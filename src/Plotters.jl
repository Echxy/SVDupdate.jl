function benchmark_time_M(m_values,n)
    # Preallocate arrays to store results
    times_f = zeros(length(m_values))
    times_g = zeros(length(m_values))
    times_h = zeros(length(m_values))
    


    for (i, m) in enumerate(m_values)
        println(i, " iter")
        
        A = rand(m,n)
        Q, S, R = svd(A) 
        a = rand(m,1)
        A = hcat(A, a)
        # variables ##################
        d_buf = zeros(size(Q,2))
        e_buf = zeros(size(Q,1))
        bQ1 = zeros(m,n+1)
        bR = zeros(n+1,n+1)
        bSVD = zeros(n+1,n+1)
        bS = zeros(n+1,n+1)
        bV = zeros(n+1,n+1)
        ############################

        """
        UpdateISVD_memory(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
        """
        GC.gc()  # Force garbage collection before timing
        times_f[i] = @elapsed UpdateISVD_memory(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)

        d_buf .= 0.0
        e_buf .= 0.0
        bQ1 .= 0.0
        bR .= 0.0
        bSVD .= 0.0
        bS .= 0.0
        bV .= 0.0

        times_g[i] = @elapsed UpdateISVD(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
       
        times_h[i] = @elapsed svd!(copy(A))  

    end

    return times_f, times_g, times_h
end


function benchmark_alloc_M(m_values,n)

    allocs_f = zeros(length(m_values))
    allocs_g = zeros(length(m_values))
    allocs_h = zeros(length(m_values))

    for (i, m) in enumerate(m_values)
        println(i, " iter")

        A = rand(m,n)
        Q, S, R = svd(A) 

        a = rand(m,1)
        A = hcat(A, a)
        # variables ##################
        d_buf = zeros(size(Q,2))
        e_buf = zeros(size(Q,1))
        bQ1 = zeros(m,n+1)
        bR = zeros(n+1,n+1)
        bSVD= zeros(n+1,n+1)
        bS = zeros(n+1,n+1)
        bV = zeros(n+1,n+1)
        ############################

        GC.gc()  # Force garbage collection before timing
        allocs_f[i] = @allocated  UpdateISVD_memory(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
        
        d_buf .= 0.0
        e_buf .= 0.0
        bQ1 .= 0.0
        bR .= 0.0
        bSVD .= 0.0
        bS .= 0.0
        bV .= 0.0
        
        allocs_g[i] = @allocated UpdateISVD(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
        allocs_h[i] = @allocated svd!(copy(A))  
    end

    return allocs_f, allocs_g, allocs_h
end




function benchmark_time_N(m,n_values)
    # Preallocate arrays to store results
    times_f = zeros(length(n_values))
    times_g = zeros(length(n_values))
    times_h = zeros(length(n_values))
    


    for (i, n) in enumerate(n_values)

        println(i, " iter")

        

        A = rand(m,n)
        Q, S, R = svd(A)  

        # Add second column incrementally
        a = rand(m,1)
        A = hcat(A, a)
        # variables ##################
        d_buf = zeros(size(Q,2))
        e_buf = zeros(size(Q,1))
        bQ1 = zeros(m,n+1)
        bR = zeros(n+1,n+1)
        bSVD = zeros(n+1,n+1)
        bS = zeros(n+1,n+1)
        bV = zeros(n+1,n+1)
        ############################

        """
        UpdateISVD_memory(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
        """
        GC.gc()  
        times_f[i] = @elapsed UpdateISVD_memory(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)

        d_buf .= 0.0
        e_buf .= 0.0
        bQ1 .= 0.0
        bR .= 0.0
        bSVD .= 0.0
        bS .= 0.0
        bV .= 0.0

        times_g[i] = @elapsed UpdateISVD(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
        
        times_h[i] = @elapsed svd!(copy(A))  

    end

    return times_f, times_g, times_h
end


function benchmark_alloc_N(m,n_values)
    
    allocs_f = zeros(length(n_values))
    allocs_g = zeros(length(n_values))
    allocs_h = zeros(length(n_values))

    for (i, n) in enumerate(n_values)
        println(i, " iter")

        

        A = rand(m,n)
        Q, S, R = svd(A)  
 
        a = rand(m,1)
        A = hcat(A, a)
        # variables ##################
        d_buf = zeros(size(Q,2))
        e_buf = zeros(size(Q,1))
        bQ1 = zeros(m,n+1)
        bR = zeros(n+1,n+1)
        bSVD = zeros(n+1,n+1)
        bS = zeros(n+1,n+1)
        bV = zeros(n+1,n+1)
        ############################

        GC.gc()  
        allocs_f[i] = @allocated  UpdateISVD_memory(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
        
        d_buf .= 0.0
        e_buf .= 0.0
        bQ1 .= 0.0
        bR .= 0.0
        bSVD .= 0.0
        bS .= 0.0
        bV .= 0.0
        
        allocs_g[i] = @allocated UpdateISVD(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
        allocs_h[i] = @allocated svd!(copy(A))  
    end
    return allocs_f, allocs_g, allocs_h
end





function plot_results(values, f, g, h, tipo,variable)
    # Combine x and y into a single metric (e.g., x * y)
    
    if tipo == "time"
        # Plot time evolution
        p1 = plot(values, [f g h], 
                xscale=:log10, yscale=:log10, 
                label=["memoria" "speed" "svd"], 
                title="Execution Time vs. $variable",
                xlabel="$variable", ylabel="Time (seconds)",
                marker=:circle, lw=2)
    else
        # Plot time evolution
        p1 = plot(values, [f g h], 
                xscale=:log10, yscale=:log10, 
                label=["memoria" "speed" "svd"], 
                title="Memory allocated vs. $variable",
                xlabel="$variable", ylabel="Memory (MB)",
                marker=:circle, lw=2)
    end
    # Combine plots
    plot(p1, size=(800, 600))
end