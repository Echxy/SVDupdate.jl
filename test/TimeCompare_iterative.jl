using SVDupdate
using TimerOutputs
using LinearAlgebra


m = 10000
n_final = 50


n = 1

reset_timer!()
tt = TimerOutput()

println("start")
println("m n ", (m,n))

A = rand(m,n)
Bb = copy(A)
Cc = copy(A)

#@timeit tt "pre-svd (pre-add row)" begin
Q, S, R = svd(A)  # Decompose first column

println(size(S))
#end
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
# argumentos
# Q S R  y la columna nueva
# d_buf e_buf
# Q1
# R1 S1 
# F V1

"""

reset_timer!(SVDupdate.tt)
@timeit SVDupdate.tt "total Speed" begin
    while n < n_final
        @timeit SVDupdate.tt "Speed Update" begin
           Q,S,R = UpdateISVD_iterative(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
        end
        #end
        n = n+1
        @timeit SVDupdate.tt "Speed alloc" begin
            if n != n_final
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
            end
        end
    end
end


println("Reconstruction error (speed): ", norm( Q*Diagonal(S)*R' - A))


Q, S, R = svd(Bb)  # Decompose first column
n = 1

a = rand(m,1)
Bb = hcat(Bb, a)
# variables ##################
d_buf = zeros(size(Q,2))
e_buf = zeros(size(Q,1))
bQ1 = zeros(m,n+1)
bR = zeros(n+1,n+1)
bSVD = zeros(n+1,n+1)
bS = zeros(n+1,n+1)
bV = zeros(n+1,n+1)


reset_timer!(SVDupdate.tt)
@timeit SVDupdate.tt "total memory" begin
    while n < n_final
        @timeit SVDupdate.tt "Memory Update" begin
           Q,S,R =  UpdateISVD_memory_iterative(Q,S,R,Bb[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
        end
        n = n + 1

        @timeit SVDupdate.tt "Memory alloc" begin
            if n != n_final
                # Add second column incrementally
                a = rand(m,1)
                Bb = hcat(Bb, a)
                # variables ##################
                d_buf = zeros(size(Q,2))
                e_buf = zeros(size(Q,1))
                bQ1 = zeros(m,n+1)
                bR = zeros(n+1,n+1)
                bSVD = zeros(n+1,n+1)
                bS = zeros(n+1,n+1)
                bV = zeros(n+1,n+1)
                ############################
            end
        end
    end
end




Cc = copy(A)
n = 1
reset_timer!(SVDupdate.tt)
@timeit SVDupdate.tt"total built-in SVD" begin
    while n < n_final

        @timeit SVDupdate.tt "built-in SVD!" begin
            F_st = svd!(copy(Cc))    
            Q_st, S_st, R_st = F_st.U, Diagonal(F_st.S), F_st.Vt
        end
        n = n+1
        if n != n_final
            a = rand(m,1)
            Cc = hcat(Cc, a)
        end
    end

end

@timeit SVDupdate.tt "Only one built-in SVD!" begin
F_st = svd!(copy(Cc))    
Q_st, S_st, R_st = F_st.U, Diagonal(F_st.S), F_st.Vt
end


#@assert Bb == A
# @assert size(Q_st) == size(bQ1)
# @assert size(S_st) == size(bS)
# @assert size(R_st) == size(bR)


println("Reconstruction error (memory): ", norm(Q*Diagonal(S)*R' - Bb))
println("Reconstruction error (Standard SVD): ", norm(Q_st * S_st * R_st - Cc))

display(SVDupdate.tt)