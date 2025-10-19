import fogbed
from fogbed.node.instance import VirtualInstance

from netfl.utils.resources import NetworkResource


class Worker(fogbed.Worker):
    def add_cluster(self, datacenter: VirtualInstance, reachable: bool = False):
        return super().add(datacenter, reachable)

    def create_cluster_link(
        self,
        node1: VirtualInstance,
        node2: VirtualInstance,
        resource: NetworkResource | None = None,
    ) -> None:
        if resource is not None:
            return super().add_link(node1, node2, **resource.link_params)

        return super().add_link(node1, node2)
