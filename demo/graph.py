import matplotlib.pyplot as plt


original_positive_accuracy = 0.3542  # Replace with actual data

compressed_positive_accuracy = 0.3890  # Replace with actual data

original_positive_noise_accuracy = 0.4021  # Replace with actual data

compressed_positive_noise_accuracy = 0.4214  # Replace with actual data

labels = ['Original', 'Compressed', 'Original with Noise', 'Compressed with Noise']

x = range(len(labels))
accuracy_values = [original_positive_accuracy, compressed_positive_accuracy, original_positive_noise_accuracy, compressed_positive_noise_accuracy]
bar_width = 0.35
plt.bar(x, accuracy_values, color=['skyblue', 'skyblue', 'skyblue', 'skyblue'], width=bar_width)

plt.xlabel('Input')
plt.ylabel('Average output')
plt.title('Negative Input on fnet-base-finetuned model with ss2 Dataset')
plt.xticks(x, labels)
plt.ylim(0, 1)  

for i, v in enumerate(accuracy_values):
    plt.text(i, v + 0.01, str(v), ha='center')

plt.tight_layout()
plt.show()
