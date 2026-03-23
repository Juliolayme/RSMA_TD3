cd RSMA_DRL

# Chạy mặc định (M=4, K=2, 300 episodes)
python3 main.py

# Tùy chỉnh tham số
python3 main.py --M 8 --K 4 --episodes 500 --channel rician

# Kênh thay đổi theo thời gian
python3 main.py --time-varying --episodes 300

# Vẽ đồ thị sau khi train xong
python3 plot_results.py
