import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean academic plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

def load_evaluation_data(filepath="evaluation/data/eval_results.json"):
    """Loads the Ragas JSON output file and calculates aggregate means."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find evaluation results at {filepath}. Please run the evaluator first.")
        
    with open(filepath, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Extract the core metrics we need
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    means = {metric: df[metric].mean() for metric in metrics if metric in df.columns}
    
    return df, means

def plot_rag_triad_radar(means, output_path="evaluation/data/ragas_radar_chart.png"):
    """Generates an academic radar/spider chart mapping the Ragas metric topology."""
    categories = ['Faithfulness', 'Answer Relevancy', 'Context Precision', 'Context Recall']
    scores = [means.get(m.lower().replace(' ', '_'), 0) for m in categories]
    
    # Radar calculations for circular closure
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    # Draw the background grid and axis lines
    plt.xticks(angles[:-1], categories, color='black', size=11)
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
    plt.ylim(0, 1.0)
    
    # Plot the system performance boundary
    ax.plot(angles, scores, color='#1f77b4', linewidth=2, linestyle='solid', label='Our System Pipeline')
    ax.fill(angles, scores, color='#1f77b4', alpha=0.25)
    
    ax.set_title("Ragas Performance Profile Matrix", va='bottom', pad=20, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"🎨 Saved Radar Chart to {output_path}")

def plot_component_performance(means, output_path="evaluation/data/ragas_components.png"):
    """Creates a grouped bar chart isolating Retrieval vs Generation metrics."""
    retrieval_metrics = {'Context Precision': means.get('context_precision', 0), 
                         'Context Recall': means.get('context_recall', 0)}
    generation_metrics = {'Faithfulness': means.get('faithfulness', 0), 
                          'Answer Relevancy': means.get('answer_relevancy', 0)}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    
    # 1. Retrieval Plot (FlashRank Re-ranker impact)
    colors_ret = ['#2ca02c', '#98df8a']
    ax1.bar(retrieval_metrics.keys(), retrieval_metrics.values(), color=colors_ret, edgecolor='grey', width=0.5)
    ax1.set_title("Retrieval Component (Search Engine Accuracy)", weight='bold', pad=15)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Ragas Score (Normalized)")
    
    # Add numerical labels over the bars
    for i, v in enumerate(retrieval_metrics.values()):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', weight='bold')
        
    # 2. Generation Plot (LLM System Prompt tracking)
    colors_gen = ['#d62728', '#ff9896']
    ax2.bar(generation_metrics.keys(), generation_metrics.values(), color=colors_gen, edgecolor='grey', width=0.5)
    ax2.set_title("Generation Component (LLM Synthesis Profile)", weight='bold', pad=15)
    
    for i, v in enumerate(generation_metrics.values()):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', weight='bold')
        
    plt.suptitle("Pipeline Module Validation Analysis", y=0.98, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"🎨 Saved Component Chart to {output_path}")

if __name__ == "__main__":
    try:
        print("📊 Extracting system metrics from raw json store...")
        df, means = load_evaluation_data()
        
        # Output console breakdown for instant validation tracking
        print("\n--- Mean Performance Metrics Summary ---")
        for metric, val in means.items():
            print(f"🔹 {metric.replace('_', ' ').title()}: {val:.4f}")
        print("----------------------------------------\n")
        
        # Build the visualizations
        plot_rag_triad_radar(means)
        plot_component_performance(means)
        print("🏁 All thesis-ready plots have been successfully synthesized and stored!")
        
    except Exception as e:
        print(f"❌ Visualization routine aborted: {e}")