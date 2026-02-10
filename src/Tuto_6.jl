using LinearAlgebra
using Plots
using SparseArrays
using KrylovKit
using Random
using ITensors

N=2
si=Vector()
for qq = 1:N
    push!(si, Index(2))
end


psi=ITensor(si)
psi[1]=1.0

psi_norm=(dag(psi)*psi)
display(psi_norm)

function buld_ghz_circuit_circuit_iten(si::Vector)::Vector
    circ=Vector()
    jj=si[1]
    ii=prime(jj)

    H = ITensor(ii, jj)
    H[ii=>1,jj=>1,]=1.0/sqrt(2)
    H[ii=>2,jj=>1,]=1.0/sqrt(2)
    H[ii=>1,jj=>2,]=1.0/sqrt(2)
    H[ii=>2,jj=>2]=-1.0/sqrt(2)
    push!(circ, H)


    for qq =1:N
        jj1=si[qq]
        ii1=prime(jj1)

        jj2=si[qq-1]
        ii2=prime(2)

        cnot=ITensor(ii1,ii2,jj1,jj2)
        cnot[ii1=>1,ii2=>1,jj1=>1,jj2=>1]=1.0
        cnot[ii1=>1,ii2=>2,jj1=>1,jj2=>2]=1.0
        cnot[ii1=>2,ii2=>1,jj1=>2jj2=>2]=1.0
        cnot[ii1=>2,ii2=>2,jj1=>2,jj2=>1]=1.0
        push!(circ, cnot)
    end
    return circ
end

circ=build_ghz_circuit_iten(si)
for gate in circ
    global psi=noprime(gate*psi)
end 


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

function build_tensor_trotter_circuit_TI(N::Integer, J::Float64, hx::Float64, dt::Float64, si::Float64)
    sm=[0 1 ; 0 0]
    sx=sm+sm'
    sz=sm'*sm-sm*sm'
    circ=Vector()

    #Trotter circ
    H12= J*kron(sz,sz)+h*kron(Id,sx)

end 