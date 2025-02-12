using SVDupdate
using TimerOutputs
using LinearAlgebra


m = 10000
n_final = 50
n = 1


println("start")
println("m n ", (m,n))

A = rand(m,n)
Bb = copy(A)
Cc = copy(A)



"""
# argumentos
# Q S R  y la columna nueva
# d_buf e_buf
# Q1
# R1 S1 
# F V1

"""

Q, S, R = svd(A)  # Decompose first column

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
bV = zeros(n+1,n+1)
############################



reset_timer!(SVDupdate.tt)

@timeit SVDupdate.tt "total Speed" begin



    while n < n_final

        
        if n == 1
        
            global Q
            global S
            global R
            global A
            global a
            global n
            global d_buf 
            global e_buf
            global bQ1 
            global bR 
            global bSVD 
            global bV

        end

        @timeit SVDupdate.tt "Speed Update" begin
           
            Q,S,R = UpdateISVD_iterative(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bSVD,bV)

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
                bV = zeros(n+1,n+1)
                ############################
            else 
                println("Reconstruction error (speed): ", norm( Q*Diagonal(S)*R' - A))
            end
        end
    end
end
display(SVDupdate.tt)

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
bV = zeros(n+1,n+1)




reset_timer!(SVDupdate.tt)
@timeit SVDupdate.tt "total memory" begin



    while n < n_final

        if n == 1
        
            global Q
            global S
            global R
            global Bb
            global a
            global n
            global d_buf 
            global e_buf
            global bQ1 
            global bR 
            global bSVD 
            global bV

        end

        @timeit SVDupdate.tt "Memory Update" begin
           Q,S,R =  UpdateISVD_memory_iterative(Q,S,R,Bb[:,n+1] ,d_buf,e_buf,bQ1,bR,bSVD,bV)
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
                bV = zeros(n+1,n+1)
                ############################
            else
                println("Reconstruction error (memory): ", norm(Q*Diagonal(S)*R' - Bb))
            end
        end
    end
end
display(SVDupdate.tt)



Cc = copy(A)
n = 1
reset_timer!(SVDupdate.tt)
@timeit SVDupdate.tt "total built-in SVD" begin
    while n < n_final

        global Cc
        global n
        global a
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


println("Reconstruction error (Standard SVD): ", norm(Q_st * S_st * R_st - Cc))

display(SVDupdate.tt)