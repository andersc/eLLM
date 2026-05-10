use std::collections::{BTreeSet, HashMap};
use std::f16;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub file_name: String,
}

#[derive(Debug)]
pub struct SafeTensorsLoader {
    model_dir: PathBuf,
    model_files: Vec<String>,
    weight_map: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct SafeTensorsIndex {
    weight_map: HashMap<String, String>,
}

impl SafeTensorsLoader {
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        if !model_dir.is_dir() {
            return Err(anyhow!(
                "model directory does not exist: {}",
                model_dir.display()
            ));
        }

        let index_path = model_dir.join("model.safetensors.index.json");
        let (model_files, weight_map) = if index_path.exists() {
            let file = File::open(&index_path)
                .with_context(|| format!("failed to open {}", index_path.display()))?;
            let index: SafeTensorsIndex = serde_json::from_reader(file)
                .with_context(|| format!("failed to parse {}", index_path.display()))?;
            let mut unique_files = BTreeSet::new();
            for file_name in index.weight_map.values() {
                unique_files.insert(file_name.clone());
            }
            (unique_files.into_iter().collect(), index.weight_map)
        } else {
            let mut model_files = Vec::new();
            for entry in std::fs::read_dir(&model_dir)
                .with_context(|| format!("failed to read {}", model_dir.display()))?
            {
                let entry = entry?;
                let file_name = entry.file_name().to_string_lossy().to_string();
                if file_name.ends_with(".safetensors") {
                    model_files.push(file_name);
                }
            }
            model_files.sort();
            (model_files, HashMap::new())
        };

        if model_files.is_empty() {
            return Err(anyhow!(
                "no safetensors files found in {}",
                model_dir.display()
            ));
        }

        for file_name in &model_files {
            let file_path = model_dir.join(file_name);
            if !file_path.is_file() {
                return Err(anyhow!(
                    "safetensors shard listed in index is missing: {}",
                    file_path.display()
                ));
            }
        }

        Ok(Self {
            model_dir,
            model_files,
            weight_map,
        })
    }

    pub fn model_files(&self) -> &[String] {
        &self.model_files
    }

    pub fn weight_map(&self) -> &HashMap<String, String> {
        &self.weight_map
    }

    pub fn tensor_infos(&self) -> Result<Vec<TensorInfo>> {
        let mut infos = Vec::new();
        for file_name in &self.model_files {
            let file_path = self.model_dir.join(file_name);
            let file = File::open(&file_path)
                .with_context(|| format!("failed to open {}", file_path.display()))?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let safetensors = SafeTensors::deserialize(&mmap)
                .with_context(|| format!("failed to parse {}", file_path.display()))?;

            for (name, tensor_view) in safetensors.tensors() {
                infos.push(TensorInfo {
                    name,
                    shape: tensor_view.shape().to_vec(),
                    dtype: tensor_view.dtype(),
                    file_name: file_name.clone(),
                });
            }
        }
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(infos)
    }

    pub fn estimated_f16_weight_bytes(&self) -> Result<u64> {
        self.tensor_infos()?
            .iter()
            .try_fold(0u64, |total, tensor| {
                let elements = tensor.shape.iter().try_fold(1u64, |product, dim| {
                    product
                        .checked_mul(*dim as u64)
                        .ok_or_else(|| anyhow!("tensor shape is too large: {}", tensor.name))
                })?;
                let bytes = elements
                    .checked_mul(std::mem::size_of::<f16>() as u64)
                    .ok_or_else(|| anyhow!("tensor is too large: {}", tensor.name))?;
                total
                    .checked_add(bytes)
                    .ok_or_else(|| anyhow!("model is too large to estimate f16 weight bytes"))
            })
    }

    pub fn load_all_weights_f16(&self) -> Result<HashMap<String, Vec<f16>>> {
        self.load_weights_f16(None)
    }

    pub fn load_selected_weights_f16<I, S>(
        &self,
        tensor_names: I,
    ) -> Result<HashMap<String, Vec<f16>>>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let requested = tensor_names
            .into_iter()
            .map(Into::into)
            .collect::<BTreeSet<_>>();
        self.load_weights_f16(Some(&requested))
    }

    fn load_weights_f16(
        &self,
        requested: Option<&BTreeSet<String>>,
    ) -> Result<HashMap<String, Vec<f16>>> {
        let mut all_weights = HashMap::new();

        for file_name in &self.model_files {
            if let Some(requested) = requested {
                let file_has_requested_tensor = self.weight_map.is_empty()
                    || requested
                        .iter()
                        .any(|name| self.weight_map.get(name) == Some(file_name));
                if !file_has_requested_tensor {
                    continue;
                }
            }

            let file_path = self.model_dir.join(file_name);
            let file = File::open(&file_path)
                .with_context(|| format!("failed to open {}", file_path.display()))?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let safetensors = SafeTensors::deserialize(&mmap)
                .with_context(|| format!("failed to parse {}", file_path.display()))?;

            for (name, tensor_view) in safetensors.tensors() {
                if requested
                    .map(|requested| !requested.contains(&name))
                    .unwrap_or(false)
                {
                    continue;
                }
                let data = tensor_to_f16(&name, tensor_view.dtype(), tensor_view.data())?;
                all_weights.insert(name, data);
            }
        }

        if let Some(requested) = requested {
            let missing = requested
                .iter()
                .filter(|name| !all_weights.contains_key(*name))
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(anyhow!("missing requested tensors: {}", missing.join(", ")));
            }
        }

        Ok(all_weights)
    }
}

fn tensor_to_f16(name: &str, dtype: Dtype, raw_data: &[u8]) -> Result<Vec<f16>> {
    match dtype {
        Dtype::F16 => Ok(raw_data
            .chunks_exact(2)
            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()),
        Dtype::BF16 => Ok(raw_data
            .chunks_exact(2)
            .map(|chunk| {
                let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from_bits((bf16_bits as u32) << 16) as f16
            })
            .collect()),
        Dtype::F32 => Ok(raw_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f16)
            .collect()),
        _ => Err(anyhow!("unsupported tensor dtype for {name}: {dtype:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, TensorView};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn loads_safetensors_index_and_selected_f16_weights() {
        let dir = temp_model_dir("loader-index");
        std::fs::create_dir_all(&dir).unwrap();

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0];
        let a_bytes = f32_bytes(&a);
        let b_bytes = f32_bytes(&b);
        let tensors = vec![
            (
                "model.language_model.embed_tokens.weight",
                TensorView::new(Dtype::F32, vec![2, 2], &a_bytes).unwrap(),
            ),
            (
                "lm_head.weight",
                TensorView::new(Dtype::F32, vec![1, 2], &b_bytes).unwrap(),
            ),
        ];
        serialize_to_file(
            tensors,
            &None,
            &dir.join("model-00001-of-00001.safetensors"),
        )
        .unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
                "metadata": {"total_size": 24},
                "weight_map": {
                    "model.language_model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                    "lm_head.weight": "model-00001-of-00001.safetensors"
                }
            }"#,
        )
        .unwrap();

        let loader = SafeTensorsLoader::new(&dir).unwrap();
        assert_eq!(loader.model_files(), &["model-00001-of-00001.safetensors"]);
        assert_eq!(loader.estimated_f16_weight_bytes().unwrap(), 12);

        let weights = loader
            .load_selected_weights_f16(["lm_head.weight".to_string()])
            .unwrap();
        assert_eq!(weights.len(), 1);
        assert_eq!(weights["lm_head.weight"][0], 5.0f32 as f16);

        let infos = loader.tensor_infos().unwrap();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].dtype, Dtype::F32);

        let _ = std::fs::remove_dir_all(dir);
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn temp_model_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ellm-{name}-{nanos}"))
    }
}
