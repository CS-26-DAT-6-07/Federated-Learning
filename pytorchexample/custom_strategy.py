import io
from flwr.serverapp.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

STRATEGY_NAME = "Tree_Custom_Strategy"

class CustomStrategy(FedAvg):
    def __init__(self, edge_groups, *args, **kwargs):  #Keep *args for safety, might not need it
        super().__init__(*args, **kwargs) 
        self.edge_groups = edge_groups

    #def configure_fit(self, server_round, arrays, config, grid): #This doesnt do anything right now. 
    #    return super().configure_fit(server_round, arrays, config, grid)

    def aggregate_fit(self, server_round, results, failures): #Happens after training
        edge_results = {0: [], 1: []}

        for client, fit_res in results:
            partid = client.node_config["partition-id"]

            for edge_id, group in self.edge_groups.items():
                if partid in group:
                    edge_results[edge_id].append((client, fit_res))
                    break

        edge_aggregates = []

        for edge_id, group_results in edge_results.items():
            if len(group_results) == 0:
                continue

            edge_arrays, edge_metrics = super().aggregate_fit(
                server_round,
                group_results,
                []
            )

            edge_aggregates.append((edge_arrays, edge_metrics))

        if not edge_aggregates:
            return None, {}
        
        #Extract arrays and weights
        arrays_list = []
        weights = []

        for arrays, metrics in edge_aggregates:
            arrays_list.append(parameters_to_ndarrays(arrays))
            weights.append(metrics["num-examples"])
        
        #Weighted average
        avg_ndarrays = []
        for layer_idx in range(len(arrays_list[0])):
            layer_sum = sum(
                w * client[layer_idx] for w, client in zip(weights, arrays_list)
            )
            avg_ndarrays.append(layer_sum / sum(weights))

        #Convert back to Flower format
        global_arrays = ndarrays_to_parameters(avg_ndarrays)

        return global_arrays, {"num-examples": sum(weights)}   