import matplotlib.pyplot as pyplot

# Dòng 04: là một hàm trả gia một cặp, fig - hình ảnh và ax- trục của hình ảnh.
# Nếu chúng ta muốn thay đổi các đặc tính của ảnh, hay lựu lại ảnh thành một tệp,
# thì fig sẽ giúp chúng ta thực hiện điều đó.

fig, ax = pyplot.subplots() # (figsize=(12,8))

# Dòng 05: Vẽ biểu đồ của mô hình dự đoán mà chúng ta có được từ Chương trình 5.1.
# Biểu đồ của chúng ta là đường thẳng màu đỏ (Prediction)

ax.plot(x_pop, f, 'r', label='Prediction')

# Dòng 06: Biểu đồ các dữ liệu chúng ta có để làm tập dữ liệu huấn luyện (Training Data)

ax.scatter(X_dacTinh, y, label='Traning Data')

# Dòng 07: hàm legend để chỉ vị trí mà các nhãn của các biểu đồ chúng ta xây dựng
# (có giá trị trong 0,...,10). Hãy xem thêm phần phụ lục.

ax.legend(loc=2)

# Dòng 08: Đặt tiêu đề cho trục hoành

ax.set_xlabel('Population in 10.000 people')

# Dòng 09: Đặt tiêu đề cho trục tung

ax.set_ylabel('Profit in 10.000 USD')

# Dòng 10: Đặt tiêu đề cho đồ thị
ax.set_title('Predicted Profit vs. Population Size')

# Dong 11: ve do thi hop (box)

duLieu.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()