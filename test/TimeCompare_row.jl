using SVDupdate
using TimerOutputs
using LinearAlgebra


m = 10000
n = 300

reset_timer!()
tt = TimerOutput()

println("start")
println("m n ", (m,n))

A = rand(m,n)
Q, S, R = svd(A)
a = rand(1,n)
B = vcat(A, a)


Q1 = zeros(m+1,n)
p = zeros(1,n)

UpdateISVD_row(Q,S,R,a,p,Q1)
Q1 .= 0.0
p .= 0.0

reset_timer!(SVDupdate.tt)
@timeit SVDupdate.tt "Speed Update" begin
    UpdateISVD_row(Q,S,R,B[m+1,:]',p,Q1)
end
display(SVDupdate.tt)

# kk = length(S)
# Ss = zeros(kk,kk)
# Ss[diagind(Ss)] .= S

println("Reconstruction error (speed): ", norm(Q1* Diagonal(S) *R' - B))

svd!(copy(B))    
reset_timer!(SVDupdate.tt)
@timeit SVDupdate.tt "built-in SVD!" begin
    F_st = svd!(copy(B))    
    Q_st, S_st, R_st = F_st.U, Diagonal(F_st.S), F_st.Vt
end
display(SVDupdate.tt)

@assert size(Q_st) == size(Q1)
@assert size(S_st) == size(Diagonal(S))
@assert size(R_st) == size(R)


println("Reconstruction error (Standard SVD): ", norm(Q_st * S_st * R_st - B))