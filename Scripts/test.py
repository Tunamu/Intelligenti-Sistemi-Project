import matplotlib.pyplot as plt

# Örnek kümeler (27 elemanlı)
agency_results = [93.06, 87.50, 93.81, 83.33, 41.67, 81.67, 93.00, 95.00, 70.00, 86.61, 91.90, 97.50, 95.00, 81.43, 69.00, 93.04, 29.17, 81.43, 63.33, 95.48, 90.50, 85.95, 94.00, 90.71, 87.62, 90.00]
rf_results = [79.82, 69.21, 73.82, 59.87, 38.67, 56.00, 89.33, 64.00, 40.62, 60.73, 54.70, 87.22, 88.53, 65.28, 47.45, 60.69, 30.83, 51.65, 41.40, 75.56, 77.92, 79.03, 56.83, 62.96, 58.70, 65.17]
x_plot_names = ["A", "B", "C", "D", "E", "F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

# İndeksler (1'den 27'ye kadar)
index = list(range(1, 27))

# Grafik oluşturma
plt.figure(figsize=(12, 6))
plt.plot(index, agency_results, marker='o', label='Agency Accuracies', color='blue')
plt.plot(index, rf_results, marker='o', label='Random Forest Accuracies', color='orange')

# Etiketler ve başlık
plt.xticks(ticks=index, labels=x_plot_names)
plt.title('Comparision Table')
plt.xlabel('Character Name')
plt.ylabel('Accuracy Value')
plt.legend()
plt.grid(True)

# Göster
plt.show()
