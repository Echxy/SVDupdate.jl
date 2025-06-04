using SVDupdate
using TimerOutputs
using LinearAlgebra


# valores iniciales
m = 1000
m_ini = 1000
m_final = 2000
# valor final de n al cual se itera
n = 200

println("start")
println("m n ", (m,n))

# una copia inicial para cada test que se hará
A = rand(m,n)
Afake = copy(A)

Q, S, R = svd(Afake)  # Decompose first column

a = rand(1,n)

Afake = vcat(Afake, a)
# variables ##################

Q1 = zeros(m+1,n)
p = zeros(1,n)

while m <= m_ini + 3

    if m == m_ini 

        global Q
        global S
        global R
        global Afake

        global Q1
        global p
        global m
        global a
    end

    Q1 = UpdateISVD_row_iterative(Q, S, R, Afake[m+1 , :]' , p, Q1)

    Q = copy(Q1)
    m += 1
    a = rand(1,n)
    Afake = vcat(Afake, a)
    Q1 = zeros(m+1,n)
    p = zeros(1,n)
end


m = m_ini

Q, S, R = svd(A)  # Decompose first column
a = rand(1,n)
A = vcat(A, a)

Q1 = zeros(m+1,n)
p = zeros(1,n)

reset_timer!(SVDupdate.tt)

@timeit SVDupdate.tt "total Speed" begin

    while m <= m_final

        if m == m_ini 

            global Q
            global S
            global R
            global A

            global Q1
            global p
            global m
            global a
        end

        @timeit SVDupdate.tt "Speed Update" begin

            Q1 = UpdateISVD_row_iterative(Q, S, R, A[m+1 , :]' , p, Q1)

        end

        if m < m_final
            Q = copy(Q1)
            m += 1
            a = rand(1,n)
            A = vcat(A, a)
            Q1 = zeros(m+1,n)
            p = zeros(1,n)
        else
            m = m+1
        end

    end
end
display(SVDupdate.tt)

println("Reconstruction error (speed): ", norm(Q1 * Diagonal(S) * R' - A))

m = m_ini
Cc = copy(A)
reset_timer!(SVDupdate.tt)

@timeit SVDupdate.tt "built-in SVD!" begin

    while m < m_final

        global Cc 
        global m 
        global a

        global Q_st
        global S_st
        global R_st
        global F_st


        @timeit SVDupdate.tt "built-in SVD!" begin
            F_st = svd!(copy(Cc))    
            Q_st, S_st, R_st = F_st.U, Diagonal(F_st.S), F_st.Vt
        end

        m += 1
        if m != m_final
            a = rand(1,n)
            Cc = vcat(Cc, a)
        else
            println("Reconstruction error (built-in SVD): ", norm(Q_st * S_st * R_st - Cc))
        end
        
    end
end
display(SVDupdate.tt)

reset_timer!(SVDupdate.tt)
@timeit SVDupdate.tt "total built-in SVD" begin
    F_st = svd!(copy(Cc))    
    Q_st, S_st, R_st = F_st.U, Diagonal(F_st.S), F_st.Vt
end

display(SVDupdate.tt)

println("Reconstruction error (total built-in SVD): ", norm(Q_st * S_st * R_st - Cc))