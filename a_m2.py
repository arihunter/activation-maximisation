import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import os
import numpy as np

def get_model_shortname(model_name):
    """Extract short model name from full path"""
    return model_name.split('/')[-1]

def maximize_neuron_activation(
    model_name="castorini/rankllama-v1-7b-lora-passage",
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    input_text="The quick brown fox",
    target_layer=-1,
    target_neuron_index=100,
    learning_rate=0.1,
    max_iterations=1000,
    lambda_smoothness=0.1,
    lambda_l2=0.1,
    epsilon=1e-4,
    patience=5,
    output_dir="results"
):
    # Create descriptive prefix for files
    model_short = get_model_shortname(model_name)
    file_prefix = f"{model_short}_layer{target_layer}_neuron{target_neuron_index}"
    
    # Create output directory with model and neuron info
    output_dir = os.path.join(output_dir, file_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval() 
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    input_embedding = model.get_input_embeddings()(input_ids).detach()
    input_embedding.requires_grad = True
    
    optimizer = torch.optim.Adam([input_embedding], lr=learning_rate)
    
    activation_history = []
    loss_history = []
    
    # Variables for tracking best result
    best_activation = float('-inf')
    best_embedding = None
    best_iteration = -1
    
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
        
        if current_activation > best_activation:
            best_activation = current_activation
            best_embedding = input_embedding.detach().clone()
            best_iteration = iteration
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print(f"Current Activation: {current_activation:.4f}")
            print(f"Best Activation: {best_activation:.4f}")
            print(f"No improvement count: {no_improvement_count}\n")
        
        if no_improvement_count >= patience:
            print(f"\nConverged after {iteration + 1} iterations!")
            print(f"Best activation {best_activation:.6f} found at iteration {best_iteration}")
            break
    
    # Save files with descriptive names
    embedding_filename = f"{file_prefix}_embedding.pt"
    plot_filename = f"{file_prefix}_history.png"
    results_filename = f"{file_prefix}_results.txt"
    activation_filename = f"{file_prefix}_activation_history.npy"
    
    # Save the best embedding
    torch.save(best_embedding, os.path.join(output_dir, embedding_filename))
    
    # Get text from best embedding
    logits = model.get_input_embeddings().weight @ best_embedding.transpose(1, 2)
    optimized_input_ids = torch.argmax(logits.transpose(1, 2), dim=-1)
    optimized_text = tokenizer.decode(optimized_input_ids[0], skip_special_tokens=True)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(activation_history)
    plt.axvline(x=best_iteration, color='r', linestyle='--', label='Best activation')
    plt.title('Activation History')
    plt.xlabel('Iteration')
    plt.ylabel('Activation')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    improvements = np.diff(activation_history)
    plt.plot(improvements)
    plt.title('Activation Improvements')
    plt.xlabel('Iteration')
    plt.ylabel('Improvement')
    plt.axhline(y=epsilon, color='r', linestyle='--', label=f'ε={epsilon}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    
    # Save detailed results
    with open(os.path.join(output_dir, results_filename), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Layer: {target_layer}\n")
        f.write(f"Neuron: {target_neuron_index}\n")
        f.write(f"Original text: {input_text}\n")
        f.write(f"Optimized text: {optimized_text}\n")
        f.write(f"Best activation: {best_activation}\n")
        f.write(f"Best iteration: {best_iteration}\n")
        f.write(f"Total iterations: {len(activation_history)}\n")
        f.write(f"Final loss: {loss_history[-1]}\n")
        f.write(f"Embedding shape: {best_embedding.shape}\n")
        f.write("\nActivation history:\n")
        for i, act in enumerate(activation_history):
            f.write(f"Iteration {i}: {act:.6f}\n")

    # Save activation history
    np.save(os.path.join(output_dir, activation_filename), np.array(activation_history))
    
    print(f"\nResults saved in: {output_dir}")
    print(f"Embedding saved as: {embedding_filename}")
    
    return {
        'best_activation': best_activation,
        'best_embedding': best_embedding,
        'best_iteration': best_iteration,
        'optimized_text': optimized_text,
        'activation_history': activation_history,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    results = maximize_neuron_activation()
