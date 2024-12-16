import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import os
import numpy as np

def maximize_neuron_activation(
    model_name="castorini/rankllama-v1-7b-lora-passage",
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    input_text="The quick brown fox",
    target_layer=-1,
    target_neuron_index=100,
    learning_rate=0.1,
    max_iterations=1000,  # Increased maximum iterations
    lambda_smoothness=0.1,
    lambda_l2=0.1,
    epsilon=1e-4,  # Convergence threshold
    patience=5,    # Number of iterations to wait before stopping
    output_dir="/project/pi_allan_umass_edu/abarjatya/results"
):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    input_embedding = model.get_input_embeddings()(input_ids).detach()
    input_embedding.requires_grad = True
    
    optimizer = torch.optim.Adam([input_embedding], lr=learning_rate)
    
    activation_history = []
    loss_history = []
    
    # Variables for convergence checking
    best_activation = float('-inf')
    no_improvement_count = 0
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        outputs = model(inputs_embeds=input_embedding, return_dict=True)
        hidden_states = outputs.hidden_states[target_layer]
        target_activation = hidden_states[0, :, target_neuron_index].mean()
        
        # Regularization
        smoothness = torch.norm(torch.diff(input_embedding, dim=1))
        l2_reg = torch.norm(input_embedding)
        
        loss = -target_activation
        loss += lambda_smoothness * smoothness
        loss += lambda_l2 * l2_reg
        
        loss.backward()
        optimizer.step()
        
        current_activation = target_activation.item()
        activation_history.append(current_activation)
        loss_history.append(loss.item())
        
        # Check for convergence
        activation_improvement = current_activation - best_activation
        
        if activation_improvement > epsilon:
            best_activation = current_activation
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print(f"Loss: {loss.item():.4f}")
            print(f"Activation: {current_activation:.4f}")
            print(f"Improvement: {activation_improvement:.6f}")
            print(f"No improvement count: {no_improvement_count}\n")
        
        # Stop if no improvement for 'patience' iterations
        if no_improvement_count >= patience:
            print(f"\nConverged after {iteration + 1} iterations!")
            print(f"No improvement greater than {epsilon} for {patience} iterations")
            break
    
    # Save results
    logits = model.get_input_embeddings().weight @ input_embedding.transpose(1, 2)
    optimized_input_ids = torch.argmax(logits.transpose(1, 2), dim=-1)
    optimized_text = tokenizer.decode(optimized_input_ids[0], skip_special_tokens=True)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot activation history
    plt.subplot(1, 2, 1)
    plt.plot(activation_history)
    plt.title('Activation History')
    plt.xlabel('Iteration')
    plt.ylabel('Activation')
    
    # Plot activation improvements
    plt.subplot(1, 2, 2)
    improvements = np.diff(activation_history)
    plt.plot(improvements)
    plt.title('Activation Improvements')
    plt.xlabel('Iteration')
    plt.ylabel('Improvement')
    plt.axhline(y=epsilon, color='r', linestyle='--', label=f'ε={epsilon}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimization_history.png')
    plt.close()
    
    # Save detailed results
    with open(f'{output_dir}/results.txt', 'w') as f:
        f.write(f"Original text: {input_text}\n")
        f.write(f"Optimized text: {optimized_text}\n")
        f.write(f"Final activation: {activation_history[-1]}\n")
        f.write(f"Final loss: {loss_history[-1]}\n")
        f.write(f"Number of iterations: {len(activation_history)}\n")
        f.write(f"Converged: {no_improvement_count >= patience}\n")
        f.write(f"Best activation: {best_activation}\n")

if __name__ == "__main__":
    maximize_neuron_activation()
