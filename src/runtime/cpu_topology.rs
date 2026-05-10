use std::collections::{BTreeMap, HashMap};
use std::env;

use core_affinity::CoreId;

const THREAD_LIMIT_ENV: &str = "ELLM_NUM_THREADS";

#[derive(Debug, Clone)]
pub struct CpuCore {
    pub logical_id: usize,
    pub core_id: Option<CoreId>,
}

#[derive(Debug, Clone)]
pub struct CpuSocket {
    pub socket_id: usize,
    pub cores: Vec<CpuCore>,
}

#[derive(Debug, Clone)]
pub struct CpuTopology {
    sockets: Vec<CpuSocket>,
}

#[derive(Debug, Clone)]
pub struct WorkerPlacement {
    pub thread_id: usize,
    pub socket_id: usize,
    pub socket_thread_id: usize,
    pub socket_thread_count: usize,
    pub core_id: Option<CoreId>,
}

impl CpuTopology {
    pub fn discover() -> Self {
        let core_ids = core_affinity::get_core_ids()
            .unwrap_or_default()
            .into_iter()
            .map(|core_id| (core_id.id, Some(core_id)))
            .collect::<Vec<_>>();
        let package_lookup = discover_physical_packages();
        Self::from_core_packages(core_ids, &package_lookup, configured_thread_limit())
    }

    pub fn from_logical_cores(
        logical_core_ids: Vec<usize>,
        package_lookup: &HashMap<usize, usize>,
        thread_limit: Option<usize>,
    ) -> Self {
        let core_ids = logical_core_ids
            .into_iter()
            .map(|logical_id| (logical_id, None))
            .collect::<Vec<_>>();
        Self::from_core_packages(core_ids, package_lookup, thread_limit)
    }

    fn from_core_packages(
        core_ids: Vec<(usize, Option<CoreId>)>,
        package_lookup: &HashMap<usize, usize>,
        thread_limit: Option<usize>,
    ) -> Self {
        if core_ids.is_empty() {
            return Self {
                sockets: vec![CpuSocket {
                    socket_id: 0,
                    cores: vec![CpuCore {
                        logical_id: 0,
                        core_id: None,
                    }],
                }],
            };
        }

        let mut grouped = BTreeMap::<usize, Vec<CpuCore>>::new();
        for (logical_id, core_id) in core_ids {
            let socket_id = package_lookup.get(&logical_id).copied().unwrap_or(0);
            grouped
                .entry(socket_id)
                .or_default()
                .push(CpuCore { logical_id, core_id });
        }

        for cores in grouped.values_mut() {
            cores.sort_by_key(|core| core.logical_id);
        }

        let sockets = grouped
            .into_iter()
            .map(|(socket_id, cores)| CpuSocket { socket_id, cores })
            .collect::<Vec<_>>();

        let sockets = match thread_limit {
            Some(limit) => limit_sockets_balanced(sockets, limit),
            None => sockets,
        };

        if sockets.iter().any(|socket| !socket.cores.is_empty()) {
            Self { sockets }
        } else {
            Self {
                sockets: vec![CpuSocket {
                    socket_id: 0,
                    cores: vec![CpuCore {
                        logical_id: 0,
                        core_id: None,
                    }],
                }],
            }
        }
    }

    pub fn worker_count(&self) -> usize {
        self.sockets.iter().map(|socket| socket.cores.len()).sum()
    }

    pub fn socket_count(&self) -> usize {
        self.sockets.len()
    }

    pub fn sockets(&self) -> &[CpuSocket] {
        &self.sockets
    }

    pub fn worker_placements(&self) -> Vec<WorkerPlacement> {
        let mut placements = Vec::with_capacity(self.worker_count());
        for socket in &self.sockets {
            let socket_thread_count = socket.cores.len();
            for (socket_thread_id, core) in socket.cores.iter().enumerate() {
                placements.push(WorkerPlacement {
                    thread_id: placements.len(),
                    socket_id: socket.socket_id,
                    socket_thread_id,
                    socket_thread_count,
                    core_id: core.core_id,
                });
            }
        }
        placements
    }
}

fn configured_thread_limit() -> Option<usize> {
    env::var(THREAD_LIMIT_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn limit_sockets_balanced(sockets: Vec<CpuSocket>, thread_limit: usize) -> Vec<CpuSocket> {
    let available = sockets.iter().map(|socket| socket.cores.len()).sum::<usize>();
    let limit = thread_limit.min(available);
    if limit == available {
        return sockets;
    }

    let mut selected = sockets
        .iter()
        .map(|socket| CpuSocket {
            socket_id: socket.socket_id,
            cores: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut next_core = vec![0usize; sockets.len()];

    while selected.iter().map(|socket| socket.cores.len()).sum::<usize>() < limit {
        let mut progressed = false;
        for (socket_index, socket) in sockets.iter().enumerate() {
            if selected.iter().map(|socket| socket.cores.len()).sum::<usize>() == limit {
                break;
            }
            let core_index = next_core[socket_index];
            if let Some(core) = socket.cores.get(core_index) {
                selected[socket_index].cores.push(core.clone());
                next_core[socket_index] += 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }

    selected
        .into_iter()
        .filter(|socket| !socket.cores.is_empty())
        .collect()
}

#[cfg(target_os = "linux")]
fn discover_physical_packages() -> HashMap<usize, usize> {
    let mut packages = HashMap::new();
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") else {
        return packages;
    };

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        let Some(cpu_id) = name.strip_prefix("cpu").and_then(|id| id.parse::<usize>().ok()) else {
            continue;
        };
        let package_path = entry.path().join("topology/physical_package_id");
        let Ok(package) = std::fs::read_to_string(package_path) else {
            continue;
        };
        if let Ok(package_id) = package.trim().parse::<usize>() {
            packages.insert(cpu_id, package_id);
        }
    }

    packages
}

#[cfg(not(target_os = "linux"))]
fn discover_physical_packages() -> HashMap<usize, usize> {
    HashMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_cores_by_socket() {
        let package_lookup = HashMap::from([(0, 0), (1, 0), (2, 1), (3, 1)]);
        let topology = CpuTopology::from_logical_cores(vec![0, 1, 2, 3], &package_lookup, None);

        assert_eq!(topology.socket_count(), 2);
        assert_eq!(topology.worker_count(), 4);
        assert_eq!(topology.sockets()[0].socket_id, 0);
        assert_eq!(topology.sockets()[0].cores.len(), 2);
        assert_eq!(topology.sockets()[1].socket_id, 1);
        assert_eq!(topology.sockets()[1].cores.len(), 2);
    }

    #[test]
    fn thread_limit_is_balanced_across_sockets() {
        let package_lookup = HashMap::from([(0, 0), (1, 0), (2, 1), (3, 1)]);
        let topology = CpuTopology::from_logical_cores(vec![0, 1, 2, 3], &package_lookup, Some(3));

        assert_eq!(topology.worker_count(), 3);
        assert_eq!(topology.sockets()[0].cores.len(), 2);
        assert_eq!(topology.sockets()[1].cores.len(), 1);

        let placements = topology.worker_placements();
        assert_eq!(placements[0].socket_id, 0);
        assert_eq!(placements[1].socket_id, 0);
        assert_eq!(placements[2].socket_id, 1);
    }

    #[test]
    fn missing_topology_falls_back_to_one_unpinned_worker() {
        let topology = CpuTopology::from_logical_cores(Vec::new(), &HashMap::new(), None);

        assert_eq!(topology.socket_count(), 1);
        assert_eq!(topology.worker_count(), 1);
        assert!(topology.worker_placements()[0].core_id.is_none());
    }
}
