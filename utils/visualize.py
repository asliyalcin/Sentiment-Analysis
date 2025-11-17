import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_pie_with_counts(df, column, title=None):
    """
    Herhangi bir kategorik kolon için:
    - Yüzde + adet gösteren pie chart üretir
    - Unique kategori sayısına göre soft pastel renk üretir
    """

    # Değer sayıları
    counts = df[column].value_counts()
    labels = counts.index
    sizes = counts.values

    # Soft pastel renk paleti (automatically sized)
    num_classes = len(labels)
    colors = sns.color_palette("pastel", num_classes)

    # Autopct içinde hem % hem sayı gösteren format
    def autopct_format(pct):
        absolute = int(round(pct/100 * sum(sizes)))
        return f"%{pct:.1f}\n({absolute})"

    # Plot
    plt.figure(figsize=(6, 4))
    plt.pie(
        sizes,
        labels=labels,
        autopct=autopct_format,
        colors=colors,
        textprops={'fontsize': 9}
    )

    if title:
        plt.title(title)
    else:
        plt.title(f"{column} Distribution")

    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def topic_stats(df, topic_col="Title", sentiment_col="sent_score", sent_class_col="sent_label"):
    grouped = df.groupby(topic_col).agg(
        n_comments=("ID", "count"),  
        mean_sentiment=(sentiment_col, "mean"),
        pos_ratio=(sent_class_col, lambda x: (x == "positive").mean()),
        neg_ratio=(sent_class_col, lambda x: (x == "negative").mean())
    ).reset_index()
    
    # en düşük ve en yüksek topicler
    grouped.sort_values("mean_sentiment")  

    return grouped
