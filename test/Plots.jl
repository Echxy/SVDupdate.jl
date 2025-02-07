using SVDupdate


m_values = 100 .+ 2 .^ (2:16) 
n = 100

n_values = 10 .+ 2 .^ (2:10)
m = 2000

times_mem, times_speed, times_svd = benchmark_time_M(m_values,n)
plot_results(m_values,times_mem, times_speed, times_svd,"time","m")

alloc_mem, alloc_speed, alloc_svd = benchmark_alloc_M(m_values,n)
plot_results(m_values, alloc_mem, alloc_speed, alloc_svd,"alloc","m")

times_mem, times_speed, times_svd = benchmark_time_N(m,n_values)
plot_results(n_values,times_mem, times_speed, times_svd,"time","n")

alloc_mem, alloc_speed, alloc_svd = benchmark_alloc_N(m,n_values)
plot_results(n_values, alloc_mem, alloc_speed, alloc_svd,"alloc","n")