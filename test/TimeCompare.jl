
using SVDupdate
using TimerOutputs
using LinearAlgebra

m = 100000
n = 300

reset_timer!()
tt = TimerOutput()

println("start")
println("m n ", (m,n))
A = rand(m,n)

Q, S, R = svd(A)  # Decompose first column

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

UpdateISVD(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)

########################
d_buf .= 0.0
e_buf .= 0.0
bQ1 .= 0.0
bR .= 0.0
bSVD .= 0.0
bS .= 0.0
bV .= 0.0
########################

reset_timer!(SVDupdate.tt)
@timeit SVDupdate.tt "Speed Update" begin
    UpdateISVD(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
end
display(SVDupdate.tt)
println("Reconstruction error (speed): ", norm(bQ1*bS*bR' - A))

########################
d_buf .= 0.0
e_buf .= 0.0
bQ1 .= 0.0
bR .= 0.0
bSVD .= 0.0
bS .= 0.0
bV .= 0.0
########################



UpdateISVD_memory(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)


########################
d_buf .= 0.0
e_buf .= 0.0
bQ1 .= 0.0
bR .= 0.0
bSVD .= 0.0
bS .= 0.0
bV .= 0.0
########################
reset_timer!(SVDupdate.tt)
@timeit SVDupdate.tt "Memory Update" begin
    UpdateISVD_memory(Q,S,R,A[:,n+1] ,d_buf,e_buf,bQ1,bR,bS,bSVD,bV)
end
display(SVDupdate.tt)


svd!(copy(A))    
reset_timer!(SVDupdate.tt)
@timeit SVDupdate.tt "built-in SVD!" begin
    F_st = svd!(copy(A))    
    Q_st, S_st, R_st = F_st.U, Diagonal(F_st.S), F_st.Vt
end
display(SVDupdate.tt)


@assert size(Q_st) == size(bQ1)
@assert size(S_st) == size(bS)
@assert size(R_st) == size(bR)



println("Reconstruction error (memory): ", norm(bQ1*bS*bR' - A))
println("Reconstruction error (Standard SVD): ", norm(Q_st * S_st * R_st - A))
