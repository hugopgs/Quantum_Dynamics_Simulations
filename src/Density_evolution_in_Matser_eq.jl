using LinearAlgebra
using Plots
using SparseArrays
using KrylovKit

# Step 1: Buld a series of lowering operators
function build_lowering_ops(sm, N)

    sms = Vector{}()
    # this is an empty vector
    # ... it will be filled with matrices as elements

    for ii = 1:N
        ldim = 2 ^ (ii - 1) # the Hilbert space dimention left of spin ii
        rdim = 2 ^ (N - ii) # the Hilbert space dimention right of spin ii
        left_id = sparse(I, ldim, ldim) # identity matrix for left Hilbert space
        right_id = sparse(I, rdim, rdim) # identity matrix for right Hilbert space
        push!(sms, kron(left_id, sm, right_id))
    end

    return sms

end


# Step 1: Buld the necessary many-body spin operators
function build_sparse_spin_ops(sm, N)

    sxs = Vector{}()
    szs = Vector{}()

    for ii = 1:N
        ldim = 2 ^ (ii - 1) # the Hilbert space dimention left of spin ii
        rdim = 2 ^ (N - ii) # the Hilbert space dimention right of spin ii
        left_id = sparse(I, ldim, ldim) # identity matrix for left Hilbert space
        right_id = sparse(I, rdim, rdim) # identity matrix for right Hilbert space
        push!(sxs, kron(left_id, sm + sm', right_id))
        push!(szs, kron(left_id, sm'*sm - sm*sm', right_id))
    end

    return sxs, szs

end




# Step 2: Build Hamiltonian
function build_sparse_hamiltonian(J, alpha, hx, sxs, szs)

    N = length(sxs)
    H = spzeros(Float64, 2^N, 2^N)

    for ii = 1:N
        H += hx .* sxs[ii]
        for jj = (ii+1):N
            H += (J / (jj-ii)^alpha) .* szs[ii] * szs[jj]
        end
    end

    return H

end

function arnoldi_exp(A, psi0, m)

    D = length(psi0)

    Q = similar(A, D, m) # the projection matrix
    h = zeros(eltype(A), m, m) # Krylov projection of A
    
    Q[:,1] = psi0   # assumed normalized
    for ii = 1:(m-1)
        psi_i = A*Q[:,ii]   
        for jj = 1:ii
            h[jj,ii] = Q[:,jj]' * psi_i
            psi_i -= h[jj,ii] .* Q[:,jj]
        end
        h[ii+1, ii] = norm(psi_i)
        Q[:,ii+1] = psi_i ./ h[ii+1, ii]
    end
 
    # now return the matrix exponential
    return Q * exp(h)[:,1]

end


function sparse_ti_simulation(N, J,alpha,hx,gamma,dt, steps, m)

    sm =[0 1; 0 0]
    sms=build_lowering_ops(sm, N)
    sxs, szs = build_sparse_spin_ops(sm, N)

    psi = zeros(ComplexF64, 2^N)
    mneel= div(4^cld(N,2)-1,3)+1
    psi[mneel] = 1.0
    y = (psi*psi')[:]

    H = build_sparse_hamiltonian(J, alpha, hx, sxs, szs)
    Heff= H
    # sm .*sqrt(gamma/2)
    for ii = 1: N 
        Heff -= im .*(gamma/2) .* (sms[ii]' * sms[ii])
    end
    A = spzeros(Float64, 4^N, 4^N)
    Id = sparse(I, 2^N, 2^N)
    A += -1im .*(kron(Id, Heff) - kron(conj(Heff)), Id)
    for ii = 1:N
        A += gamma .* kron(conj(sms[ii]), sms[ii])
    end
    display(A)



    return nothing
end


N = 7 # number of spins
m=20# dimension of Krylov space
J = 1 # defines energy/time units
alpha = 3 # interaction range
hx = 2# transverse field

dt = 0.1 # time step for plotting
steps =101

@time out_sz, out_szsz =  sparse_ti_simulation(N, J, alpha, hx,0.2, dt, steps, m)