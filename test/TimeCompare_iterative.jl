using SVDupdate
using TimerOutputs
using LinearAlgebra


# valores iniciales
m = 10000
# valor final de n al cual se itera
n_final = 50
n = 1

println("start")
println("m n ", (m,n))


# una copia inicial para cada test que se hará
A = rand(m,n)
Afake = copy(A)
Bb = copy(A)
Cc = copy(A)

################################
# Afake es para iterar 1 vez y cargar las funciones y los globals
Q, S, R = svd(Afake)  # Decompose first column

a = rand(m,1)
Afake = hcat(Afake, a)
# variables ##################
d_buf = zeros(size(Q,2))
e_buf = zeros(size(Q,1))
bQ1 = zeros(m,n+1)
bR = zeros(n+1,n+1)
bSVD = zeros(n+1,n+1)
bV = zeros(n+1,n+1)
############################

# itero solo 3 veces
while n < 3 

    # defino las globales solo 1 vez
    if n == 1
        
        global Q
        global S
        global R
        global Afake
        global a
        global n
        global d_buf 
        global e_buf
        global bQ1 
        global bR 
        global bSVD 
        global bV

    end    
    
    # itero
    Q,S,R = UpdateISVD_iterative(Q,S,R,Afake[:,n+1] ,d_buf,e_buf,bQ1,bR,bSVD,bV)
    
    # incremento
    n = n+1
    a = rand(m,1)
    Afake = hcat(Afake, a)
    # variables ##################
    d_buf = zeros(size(Q,2))
    e_buf = zeros(size(Q,1))
    bQ1 = zeros(m,n+1)
    bR = zeros(n+1,n+1)
    bSVD = zeros(n+1,n+1)
    bV = zeros(n+1,n+1)
    ############################
  
end


# reseteo n
n = 1
Q, S, R = svd(A)  

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
      

        n = n+1
        @timeit SVDupdate.tt "Speed alloc" begin

            # solo hago el append hasta las n_final iteraciones, sino queda A con una columna de más
            if n != n_final
  
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

n = 1
Q, S, R = svd(Bb) 

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


n = 1
Cc = copy(A)

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