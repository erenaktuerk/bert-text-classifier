import matplotlib.pyplot as plt
import seaborn as sns

def create_sentiment_distribution_plot(df):
    """
    Creates a plot showing the distribution of sentiment labels.

    Args:
        df (pd.DataFrame): The dataset with sentiment labels.
    """
    # Set the figure size for the plot
    plt.figure(figsize=(8, 6))

    # Create a count plot for the 'sentiment' column from the dataset
    sns.countplot(x='sentiment', data=df)  # 'sentiment' remains as in the raw data

    # Set the title and labels of the plot
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    # Display the plot and wait until the window is closed before continuing the execution
    plt.show()

def create_wordcloud(df):
    """
    Generates a word cloud from the 'review' column.

    Args:
        df (pd.DataFrame): The dataset with reviews.
    """
    # Import the WordCloud package inside the function
    from wordcloud import WordCloud

    # Concatenate all reviews into a single string
    text = ' '.join(df['review'])  # 'review' remains as in the raw data

    # Create a word cloud from the concatenated reviews
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=1000, min_font_size=10).generate(text)

    # Set the figure size for the word cloud display
    plt.figure(figsize=(10, 8))

    # Display the word cloud without axes
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Show the word cloud and wait until the window is closed before continuing the execution
    plt.show()