"""
Graph Network-based Physics Simulator.

This module implements the network framework structured as:

    ENCODER:   2 MLPs to build latent representations for nodes and edges (embeddings)
    PROCESSOR: M Graph Networks (Interaction Networks) to propagate information
    DECODER:   1 MLP to predict the accelerations of the particles at the next time step

References:
    "Learning to Simulate Complex Physics with Graph Networks", Sanchez-Gonzalez et al.
"""

import torch
from torch import nn
from torch_scatter import scatter_add


class MLP(nn.Module):
    "Standard building block used everywhere in the architecture."

    def __init__(
        self, input_size, hidden_size, output_size, num_layers, layer_norm=True
    ):
        """
        Args:
            input_size: Dimension of the input vector.
            hidden_size: Dimension of the hidden layers.
            output_size: Dimension of the output vector.
            num_layers: Total number of linear layers.
            layer_norm: If True, applies Layer Normalization (stabilizes training).
        """
        super().__init__()

        layers = []
        in_dim = input_size

        # Creating hidden layers: Linear -> ReLU
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size

        # Adding the output layer without activation function
        layers.append(nn.Linear(hidden_size, output_size))

        # Including LayerNorm to prevent vanishing/exploding gradients
        if layer_norm:
            layers.append(nn.LayerNorm(output_size))

        # Stacking everything in a Sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, input_data):
        """
        Arg:
            input_data Shape: [Batch_Size, input_size]

        """
        return self.network(input_data)


class InteractionNetwork(nn.Module):
    """
    This is the core of the PROCESSOR. It performs one step of Message Passing.
    It calculates interactions between particles (edges) and updates particle states (nodes).
    """

    def __init__(self, hidden_size, num_layers_mlp):
        super().__init__()

        # MLP for Edge Update (\phi^e):
        # Input: [Current Edge_Attr, Sender_Node, Receiver_Node] so we have 3 * hidden_size input neurons
        # Output: updated edge features
        self.edges_mlp = MLP(hidden_size * 3, hidden_size, hidden_size, num_layers_mlp)

        # MLP for Node Update (\phi^v):
        # Input: [Current node features , Aggregated Message] so we have 2 * hidden_size input neurons
        # Output: Updated node features
        self.node_mlp = MLP(hidden_size * 2, hidden_size, hidden_size, num_layers_mlp)

    def forward(self, input_features, edge_index, edge_attr):
        """
        Args:
            input_features: Node features [Num_Nodes, Latent_Dim]
            edge_index: Graph connectivity [2, Num_Edges] (Senders, Receivers)
            edge_attr: Edge features [Num_Edges, Latent_Dim]
        """

        # Decompose the graph connectivity
        senders, receivers = edge_index

        # Gather features for senders and receivers
        # *_features shape: [Num_Edges, Latent_Dim]
        senders_features = input_features[senders]
        receivers_features = input_features[receivers]

        # Prepare input for Edge MLP
        # we concatenate: current edge attributes + sender features + receiver features
        # Shape: [Num_Edges, Latent_Dim * 3]
        edge_inputs = torch.cat(
            [edge_attr, senders_features, receivers_features], dim=1
        )

        # Update Edge Features
        # update_edge_features is the "message" sent along the edge
        # Shape: [Num_Edges, Latent_Dim]
        update_edge_features = self.edges_mlp(edge_inputs)

        # Aggregate Messages
        # we sum all messages (update_edge_features) directed to the same receiver.
        # Shape: [Num_Nodes, Latent_Dim]
        aggr_message = scatter_add(
            update_edge_features, receivers, dim=0, dim_size=input_features.size(0)
        )

        # Prepare input for Node MLP
        # we concatenate: current node features + aggregated incoming messages
        # Shape: [Num_Nodes, Latent_Dim * 2]
        node_inputs = torch.cat([input_features, aggr_message], dim=1)

        # Update Node Features
        # Shape: [Num_Nodes, Latent_Dim]
        update_node_features = self.node_mlp(node_inputs)

        return update_node_features, update_edge_features


class EncoderProcessorDecoder(nn.Module):
    """The full GN simulator architecture"""

    def __init__(
        self,
        input_node_dim,
        input_edge_dim,
        latent_size,
        num_layers_mlp,
        num_message_passing_steps,
        output_size,
    ):
        """
        Args:
            input_node_dim: Number of input neurons of node_encoder MLP
            input_edge_dim: Number of input neurons of edge_encoder MLP
            latent_size: Number of hidden neurons in Linear Layers
            num_layers_mlp: Number of Linear Layers + ReLU
            num_message_passing_steps: Number of InteractionNetworks in PROCESSOR
            output_size: Number of output neurons in MLP
        """

        super().__init__()

        self.num_steps = num_message_passing_steps

        # ENCODER
        # Projects raw inputs (e.g. velocity, distances, type) into a latent space of size 128
        self.node_encoder = MLP(
            input_node_dim, latent_size, latent_size, num_layers_mlp
        )

        self.edge_encoder = MLP(
            input_edge_dim, latent_size, latent_size, num_layers_mlp
        )

        # PROCESSOR
        # A stack of M independent InteractionNetworks
        self.processor_layers = nn.ModuleList(
            [
                InteractionNetwork(latent_size, num_layers_mlp)
                for _ in range(self.num_steps)
            ]
        )

        # DECODER
        # Projects the final latent node representation to the physical output : acceleration
        # note tha Decoder doesn't have LayerNorm at the end to allow unconstrained values of accelerations
        self.decoder = MLP(
            latent_size, latent_size, output_size, num_layers_mlp, layer_norm=False
        )

    def forward(self, node_input_data, edge_index, edge_input_data):
        """
        Forward pass of the simulator.

        Args:
            node_input_data: Raw node features (e.g. velocities) [Num_Nodes, Input_Node_Dim]
            edge_index: Coordinate List [2, Num_Edges]
            edge_input_data: Raw edge features (e.g., relative dist) [Num_Edges, Input_Edge_Dim]
        """

        # ENCODE: Map raw data to latent space [Num_*, Latent_size]
        node_features = self.node_encoder(node_input_data)
        edge_features = self.edge_encoder(edge_input_data)

        # PROCESS: Iterative Message Passing with Residual Connections
        # we calculated the deltas rather than the new value and then update the current
        for layer in self.processor_layers:

            # Calculate the updates (deltas)
            delta_node, delta_edge = layer(node_features, edge_index, edge_features)

            # Apply Residual Connection
            # x_new = x_old + delta
            node_features += delta_node
            edge_features += delta_edge

        # DECODE: Map latent features to physical quantity: acceleration
        # note that we only decode the nodes, as we are interested in particle (node) motion.
        output = self.decoder(node_features)

        return output
