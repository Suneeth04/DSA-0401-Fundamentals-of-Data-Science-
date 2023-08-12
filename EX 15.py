import numpy as np
def calculate_likes_frequency_distribution(likes):
    unique_likes, frequency = np.unique(likes, return_counts=True)
    likes_freq_dist = dict(zip(unique_likes, frequency))
    return likes_freq_dist
if __name__ == "__main__":
    n = int(input("Enter the number of posts: "))
    likes = []
    for i in range(n):
        likes_count = int(input(f"Enter the number of likes for post {i + 1}: "))
        likes.append(likes_count)
    likes_frequency_distribution = calculate_likes_frequency_distribution(likes)

    print("\nFrequency Distribution of Likes:")
    for likes_count, frequency in likes_frequency_distribution.items():
        print(f"{likes_count} likes: {frequency} posts")
