import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_pie_with_counts(df, column, title=None):
    """
    For any categorical column:
    - Generates a pie chart showing both percentage and counts
    - Automatically creates a soft pastel color palette based on the number of unique categories
    """

    # Value counts
    counts = df[column].value_counts()
    labels = counts.index
    sizes = counts.values

    # Soft pastel color palette (automatically sized)
    num_classes = len(labels)
    colors = sns.color_palette("pastel", num_classes)

    # Autopct formatter: shows % and absolute count
    def autopct_format(pct):
        absolute = int(round(pct / 100 * sum(sizes)))
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
    """
    Computes topic-based statistics:
    - Number of comments
    - Mean sentiment score
    - Positive ratio
    - Negative ratio
    """
    grouped = df.groupby(topic_col).agg(
        n_comments=("ID", "count"),
        mean_sentiment=(sentiment_col, "mean"),
        pos_ratio=(sent_class_col, lambda x: (x == "positive").mean()),
        neg_ratio=(sent_class_col, lambda x: (x == "negative").mean())
    ).reset_index()

    # Sort topics (optional) by lowest sentiment
    grouped.sort_values("mean_sentiment")

    return grouped
