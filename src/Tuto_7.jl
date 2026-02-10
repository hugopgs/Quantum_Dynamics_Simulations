# MPS code

using LinearAlgebra
using Plots
using SparseArrays
using KrylovKit
using Random
using ITensors

function create_exp_tensor_gate_1(inputleg1::Index, gate::Matrix)
    jj1=inputleg1
    ii1=prime(jj1)
    U= ITensor(exp(gate), ii1, jj1)
    return U 
end 

function create_exp_tensor_gate_2(inputleg1::Index, inputleg2::Index, gate::Matrix)
    jj1=inputleg1
    ii1=prime(jj1)

    jj2=inputleg2
    ii2=prime(jj2)

    A=reshape(exp(gate), (dim(ii2),dim(ii1),dim(jj2),dim(jj1)))
    U =ITensor(A, ii2,ii1,jj2,jj1)
    return U 
end 


# # Case of general J 
# function build_tensor_trotter_circuit_TI(J::Matrix, hx::Float64, N::Integer, dt::Float64, si::Vector)::Vector
#     sm = [0 1; 0 0]
#     sx = sm + sm'
#     sz = sm'*sm - sm*sm'
#     circ = Vector()
#     for jj in 2:N
#         for ii in 1:(jj-1)
#             push!(circ, create_exp_tensor_gate2(si[ii], si[jj], (-im * (dt/2) * J[ii, jj]) .* kron(sz, sz)))
#         end
#     end
#     for ii in 1:N
#         push!(circ, create_exp_tensor_gate1(si[ii], (-im * (dt/2) * hx) .* sx))
#     end
# end

# # Case of Jij = J / ((j-i)^alpha)
# function build_tensor_trotter_circuit_TI(N::Integer, J::Float64, alpha::Float64, hx::Float64, dt::Float64, si::Vector)::Vector
#     sm = [0 1; 0 0]
#     sx = sm + sm'
#     sz = sm'*sm - sm*sm'
#     circ = Vector()
#     for jj in 2:N
#         for ii in 1:(jj-1)
#             push!(circ, create_exp_tensor_gate2(si[ii], si[jj], (-im * (dt/2) * J / ((jj-ii)^alpha)) .* kron(sz, sz)))
#         end
#     end
#     for ii in 1:N
#         push!(circ, create_exp_tensor_gate1(si[ii], (-im * (dt/2) * hx) .* sx))
#     end
#     return vcat(circ, reverse(circ)) # sweep plus a reversed sweep (2nd order Trotterisation)
# end

# Case of nearest neighbours
function build_tensor_trotter_circuit_TI(N::Integer, J::Float64, hx::Float64, dt::Float64, si::Vector)::Vector
    
    sm = [0 1; 0 0]
    sx = sm + sm'
    sz = sm'*sm - sm*sm'
    circ = Vector()

    for jj in 2:N
        ii = jj - 1
        push!(circ, create_exp_tensor_gate2(si[ii], si[jj], (-im * (dt/2) * J) .* kron(sz, sz)))
    end

    for ii in 1:N
        push!(circ, create_exp_tensor_gate1(si[ii], (-im * (dt/2) * hx) .* sx))
    end

    return vcat(circ, reverse(circ)) # sweep plus a reversed sweep (2nd order Trotterisation)
end

mutable struct mps
    N::Int
    chi::Int

    gams::Vector{ITensor}
    lams::Vector{ITensor}

    qs::Vector{Index}
    lchis::Vector{Index}
    rchis::Vector{Index}

    function mps(N, chi)
        gams = Vector{ITensor}(undef, N)
        lams = Vector{ITensor}(undef, N+1)

        qs = Vector{Index}(undef, N)
        lchis = Vector{Index}(undef, N+1)
        rchis = Vector{Index}(undef, N+1)

        new(N, chi, gams, lams, qs, lchis, rchis)
    end

end

function zero_state(N, chi)
    psi=mps(N,chi)
    for qq=1:N
        psi.qs[qq]=Index(2, "q$(qq)")
    end

    for bb=1:N+1
        psi.lchis[bb]=Index(1, "lchis$(bb)")
        psi.rchis[bb]=Index(1, "rchis$(bb)")
    end

    for bb=1:(N+1)
        psi.lams[bb]=diagITensor(psi.lchis[bb],psi.rchis[bb])
        psi.lams[bb][1]=1.0
    end

    for qq=1:N
        psi.gams[qq]=ITensor(psi.rchis[qq], psi.qs[qq], psi.lchis[qq+1])
        psi.gams[qq][1]=1.0
    end
    return psi
end


function apply_1s_gate!(gate::ITensor, psi)
    qind = unique(noprime(inds(gate)))[1]
    qq = findfirst(isequal(qind), psi.qs)
    psi.gams[qq] = noprime(gate * psi.gams[qq])
    
    return nothing
end


function apply_2s_gate!(gate::ITensor, psi)
    qind1 = unique(noprime(inds(gate)))[1]
    qind2 = unique(noprime(inds(gate)))[2]
    
    qq = min(findfirst(isequal(qind1), psi.qs),findfirst(isequal(qind2), psi.qs))
    theta= psi.lams[qq]*psi.gams[qq]*psi.lams[qq+1]*psi.gams[qq+1]*psi.lams[qq+2]
    theta *= gate
    noprime!(theta)

    psi.gams[qq], psi.lams[qq+1], psi.gams[qq+1], _, psi.lchis[qq+1], psi.rchis[qq+1] = svd(theta, psi.lchis[qq], psi.qs[qq]; maxdim=psi.chi, cutoff=1.e-12, lefttags="lchi$(qq+1)", righttags="rchi$(qq+1)")


    return nothing
end







function ghz_mps(N, chi)
    psi=zero_state(N,chi)

    ii=psi.qs[1]
    jj=prime(ii)

    H = ITensor(ii, jj)
    H[ii=>1,jj=>1,]=1.0/sqrt(2)
    H[ii=>2,jj=>1,]=1.0/sqrt(2)
    H[ii=>1,jj=>2,]=1.0/sqrt(2)
    H[ii=>2,jj=>2]=-1.0/sqrt(2)
    

    apply_1s_gate!(H, psi)

    for qq =1:N-1
        jj1=si[qq]
        ii1=prime(jj1)

        jj2=si[qq-1]
        ii2=prime(2)

        cnot=ITensor(ii1,ii2,jj1,jj2)
        cnot[ii1=>1,ii2=>1,jj1=>1,jj2=>1]=1.0
        cnot[ii1=>1,ii2=>2,jj1=>1,jj2=>2]=1.0
        cnot[ii1=>2,ii2=>1,jj1=>2jj2=>2]=1.0
        cnot[ii1=>2,ii2=>2,jj1=>2,jj2=>1]=1.0

        apply_2s_gate!(cnot, psi)
    end
    return circ
end



