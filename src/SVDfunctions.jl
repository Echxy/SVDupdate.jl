tt = TimerOutput()

function mini_svd!(bSVD,S,v,p,V)

    n = length(S)
    @inbounds @views begin

        bSVD[diagind(bSVD)] .= [S; p].^2    
        bSVD[1:n, n+1] .= S .* v
        bSVD[n+1, 1:n] .= S .* v
        bSVD[n+1, n+1] += dot(v, v)
    end

    @timeit tt "eigen-value" begin
        ss, V1 = LinearAlgebra.LAPACK.syevd!('V', 'L', bSVD) 
    end

    o = sortperm(ss, rev=true) 
    permute!(ss,o)
    V .= @view V1[:,o]
    ss .= sqrt.(ss) 
    
    @timeit tt "U asignations" begin

        @inbounds @views begin
            bSVD[1:n, :] .= S .* V[1:n, :]
            mul!(bSVD[1:n, :], v, V[n+1, :]', 1.0, 1.0)
            bSVD[n+1, :] = p .* V[n+1, :]'
            bSVD ./= ss'
        end

    end

    return ss
end


function memory_svd!(bSVD,S,v,p,V)

    n = length(S)
    
    
    @inbounds @views begin
        bSVD[diagind(bSVD)] .= [S; p].^2
        bSVD[1:n, n+1] .= S .* v
        bSVD[n+1, 1:n] .= S .* v
        bSVD[n+1, n+1] += dot(v, v) 
    end


    @timeit tt "eigen-value" begin
        ss, V1 = LinearAlgebra.LAPACK.syev!('V', 'L', bSVD) # mejor en memoria
    end
    
  
    o = sortperm(ss, rev=true) 
    permute!(ss,o)
    V .= @view V1[:,o]
    @. ss .= sqrt.(ss)     


    @timeit tt "U asignations" begin

        @inbounds @views begin
            bSVD[1:n, :] .= S .* V[1:n, :]
            mul!(bSVD[1:n, :], v, V[n+1, :]', 1.0, 1.0)
            bSVD[n+1, :] = p .* V[n+1, :]'
            bSVD ./= ss'
        end
    end

    return ss
end


function UpdateISVD(Q, S, R, u,
    d,
    e,Q1,R1,S1,U1,V1)

    @timeit tt  "normalize" begin

        k = length(S)
        mul!(d, Q', u)
        mul!(e, Q, d)
        @. e = u - e
        p = norm(e)
        e ./= p


        for j in 1:size(Q, 2)
            qj = @view Q[:, j]
            α = dot(e, qj)
            BLAS.axpy!(-α, qj, e)    
        end
        normalize!(e)

    end
    

    """
    mini_svd!(U1,S,v,p,V1)
    return S
    """
    
    @timeit tt "svd_step" begin
        AS = mini_svd!(U1,S,d,p,V1)
    end

    @timeit tt "U multiply" begin
        @inbounds @views begin
            S1[diagind(S1)] .= AS
            R1[k+1, :] .= V1[k+1:k+1,:]'
            mul!(view(R1, 1:k, :), R,view(V1, 1:k, :))
            mul!(Q1, Q, view(U1, 1:k, :))    
            mul!(Q1, e, view(U1, k+1:k+1, :), 1.0, 1.0)  
        end
    end     
end



function UpdateISVD_memory(Q, S, R, u,
    d,
    e,Q1,R1,S1,U1,V1)


    @timeit tt "normalize" begin

        k = length(S)
        mul!(d, Q', u)
        mul!(e, Q, d)
        @. e = u - e
        p = norm(e)
        e ./= p


        for j in 1:size(Q, 2)
            qj = @view Q[:, j]
            α = dot(e, qj)
            BLAS.axpy!(-α, qj, e)    
        end
        normalize!(e)

    end
    """
    memory_svd!(U1,S,v,p)
    return S 

    """
    
    @timeit tt "svd_step" begin
        AS = memory_svd!(U1,S,d,p,V1)
    end 
    
    @timeit tt "U multiply" begin

        @inbounds @views begin
            S1[diagind(S1)] .= AS
            R1[k+1, :] .= V1[k+1:k+1,:]'
            mul!(view(R1, 1:k, :), R,view(V1, 1:k, :))
            mul!(Q1, Q, view(U1, 1:k, :))    
            mul!(Q1, e, view(U1, k+1:k+1, :), 1.0, 1.0) 
        end
        
    end
    
end



function UpdateISVD_memory_iterative(Q, S, R, u,
    d,
    e,Q1,R1,U1,V1)


    k = length(S)

    mul!(d, Q', u)
    mul!(e, Q, d)
    @. e = u - e
    p = norm(e)
    e ./= p


    for j in 1:size(Q, 2)
        qj = @view Q[:, j]
        α = dot(e, qj)
        BLAS.axpy!(-α, qj, e)    
    end
    normalize!(e)

    """
    memory_svd!(U1,S,v,p)
    return S 

    """
    
    AS = memory_svd!(U1,S,d,p,V1)
            
    @inbounds @views begin
        #S1[diagind(S1)] .= AS
        R1[k+1, :] .= V1[k+1:k+1,:]'
        mul!(view(R1, 1:k, :), R,view(V1, 1:k, :))
        mul!(Q1, Q, view(U1, 1:k, :))    
        mul!(Q1, e, view(U1, k+1:k+1, :), 1.0, 1.0) 
    end
    return Q1 , AS, R1
end



function UpdateISVD_iterative(Q, S, R, u,
    d,
    e,Q1,R1,U1,V1)

    k = length(S)

    mul!(d, Q', u)
    mul!(e, Q, d)
    @. e = u - e
    p = norm(e)
    e ./= p


    for j in 1:size(Q, 2)
        qj = @view Q[:, j]
        α = dot(e, qj)
        BLAS.axpy!(-α, qj, e)    
    end
    normalize!(e)
    

    """
    mini_svd!(U1,S,v,p,V1)
    return S
    """
    
    AS = mini_svd!(U1,S,d,p,V1)
        
    @inbounds @views begin
        #S1[diagind(S1)] .= AS
        R1[k+1, :] .= V1[k+1:k+1,:]'
        mul!(view(R1, 1:k, :), R,view(V1, 1:k, :))
        mul!(Q1, Q, view(U1, 1:k, :))    
        mul!(Q1, e, view(U1, k+1:k+1, :), 1.0, 1.0)  
    end
    return Q1 , AS, R1
end 