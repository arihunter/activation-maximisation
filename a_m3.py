import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import os
import numpy as np


def get_model_shortname(model_name):
    return model_name.split("/")[-1]


def smoothness_regularization(hidden_state):
    return torch.norm(torch.diff(hidden_state, dim=1))


def l2_regularization(hidden_state):
    return torch.norm(hidden_state)


def maximize_neuron_activation_hidden_state(
    model_name="castorini/rankllama-v1-7b-lora-passage",
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    input_text="The quick brown fox",
    target_layer=20,  # Layer l
    target_neuron_index=100,
    learning_rate=0.1,
    max_iterations=1000,
    lambda_smoothness=0.1,
    lambda_l2=0.1,
    epsilon=1e-4,
    patience=5,
    output_dir="results",
):
    # Create descriptive prefix for files
    model_short = get_model_shortname(model_name)
    file_prefix = f"{model_short}_targetlayer{target_layer}_neuron{target_neuron_index}_hidden"
    output_dir = os.path.join(output_dir, file_prefix)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True
    ).to(device)
    model.eval()  # Set to evaluation mode

    # Get initial hidden states up to layer l-1
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Get the hidden state from layer l-1
        prev_hidden = hidden_states[target_layer - 1].detach()

    # Make hidden state optimizable
    hidden_state = prev_hidden.clone()
    hidden_state.requires_grad = True

    # Setup optimizer
    optimizer = torch.optim.Adam([hidden_state], lr=learning_rate)

    activation_history = []
    loss_history = []
    smoothness_history = []
    l2_history = []

    # Variables for tracking best result
    best_activation = float("-inf")
    best_hidden_state = None
    best_iteration = -1

    def forward_from_hidden_wrong(hidden_state, target_layer):
        """Forward pass from a specific hidden state through remaining layers"""
        current_hidden = hidden_state

        # Get the target transformer layer
        if hasattr(model, "transformer"):
            layer = model.transformer.h[target_layer]
        else:
            layer = model.model.layers[target_layer]

        # Forward through the target layer
        outputs = layer(current_hidden)
        return outputs[0]  # Return the hidden state output

    def forward_from_hidden(hidden_state, target_layer):
        """Forward pass from a specific hidden state through remaining layers"""
        batch_size, seq_length = hidden_state.shape[:2]

        # Create position IDs (1, seq_length)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=hidden_state.device
        ).unsqueeze(0)

        # Get the target transformer layer
        if hasattr(model, "transformer"):
            layer = model.transformer.h[target_layer]
        else:
            layer = model.model.layers[target_layer]

        # Forward through the target layer with position IDs
        outputs = layer(
            hidden_state,
            position_ids=position_ids,
            attention_mask=None,  # Not needed for single sequence
            use_cache=False,
        )
        return outputs[0]  # Return the hidden state output

    for iteration in range(max_iterations):
        optimizer.zero_grad()

        # Forward pass from hidden state through layer l
        next_hidden = forward_from_hidden(hidden_state, target_layer)
        target_activation = next_hidden[0, :, target_neuron_index].mean()

        # Calculate regularization terms
        smoothness = smoothness_regularization(hidden_state)
        l2_reg = l2_regularization(hidden_state)

        # Compute loss with regularization
        loss = -target_activation
        loss += lambda_smoothness * smoothness
        loss += lambda_l2 * l2_reg

        loss.backward()
        optimizer.step()

        # Record history
        current_activation = target_activation.item()
        activation_history.append(current_activation)
        loss_history.append(loss.item())
        smoothness_history.append(smoothness.item())
        l2_history.append(l2_reg.item())

        if current_activation > best_activation:
            best_activation = current_activation
            best_hidden_state = hidden_state.detach().clone()
            best_iteration = iteration
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print(f"Current Activation: {current_activation:.4f}")
            print(f"Best Activation: {best_activation:.4f}")
            print(f"Smoothness: {smoothness.item():.4f}")
            print(f"L2: {l2_reg.item():.4f}")
            print(f"No improvement count: {no_improvement_count}\n")

        if no_improvement_count >= patience:
            print(f"\nConverged after {iteration + 1} iterations!")
            print(
                f"Best activation {best_activation:.6f} found at iteration {best_iteration}"
            )
            break

    # Save files with descriptive names
    hidden_filename = f"{file_prefix}_hidden_state.pt"
    plot_filename = f"{file_prefix}_history.png"
    results_filename = f"{file_prefix}_results.txt"
    activation_filename = f"{file_prefix}_activation_history.npy"

    # Save the best hidden state and metadata
    torch.save(
        {
            "hidden_state": best_hidden_state,
            "original_hidden": prev_hidden,
            "layer": target_layer,
            "neuron": target_neuron_index,
            "activation": best_activation,
            "iteration": best_iteration,
        },
        os.path.join(output_dir, hidden_filename),
    )

    # Plot results
    plt.figure(figsize=(15, 10))

    # Plot activation history
    plt.subplot(2, 2, 1)
    plt.plot(activation_history)
    plt.axvline(
        x=best_iteration, color="r", linestyle="--", label="Best activation"
    )
    plt.title("Activation History")
    plt.xlabel("Iteration")
    plt.ylabel("Activation")
    plt.legend()

    # Plot loss history
    plt.subplot(2, 2, 2)
    plt.plot(loss_history)
    plt.title("Loss History")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    # Plot smoothness history
    plt.subplot(2, 2, 3)
    plt.plot(smoothness_history)
    plt.title("Smoothness History")
    plt.xlabel("Iteration")
    plt.ylabel("Smoothness")

    # Plot L2 history
    plt.subplot(2, 2, 4)
    plt.plot(l2_history)
    plt.title("L2 Norm History")
    plt.xlabel("Iteration")
    plt.ylabel("L2 Norm")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

    # Save detailed results
    with open(os.path.join(output_dir, results_filename), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Input text: {input_text}\n")
        f.write(f"Target layer: {target_layer}\n")
        f.write(f"Target neuron: {target_neuron_index}\n")
        f.write(f"Best activation: {best_activation}\n")
        f.write(f"Best iteration: {best_iteration}\n")
        f.write(f"Total iterations: {len(activation_history)}\n")
        f.write(f"Final smoothness: {smoothness_history[-1]}\n")
        f.write(f"Final L2 norm: {l2_history[-1]}\n")
        f.write(f"Hidden state shape: {best_hidden_state.shape}\n")
        f.write("\nActivation history:\n")
        for i, act in enumerate(activation_history):
            f.write(f"Iteration {i}: {act:.6f}\n")

    # Save histories
    np.savez(
        os.path.join(output_dir, activation_filename),
        activation=np.array(activation_history),
        loss=np.array(loss_history),
        smoothness=np.array(smoothness_history),
        l2=np.array(l2_history),
    )

    print(f"\nResults saved in: {output_dir}")
    print(f"Hidden state saved as: {hidden_filename}")

    return {
        "best_activation": best_activation,
        "best_hidden_state": best_hidden_state,
        "original_hidden_state": prev_hidden,
        "best_iteration": best_iteration,
        "activation_history": activation_history,
        "smoothness_history": smoothness_history,
        "l2_history": l2_history,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    results = maximize_neuron_activation_hidden_state()

