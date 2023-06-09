import flwr as fl

fl.server.start_server("[::]:8080", config={"num_rounds": 3})
