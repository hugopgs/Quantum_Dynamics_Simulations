using LinearAlgebra
using Plots
using SparseArrays
using KrylovKit
using Random
# using LaTeXStrings

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

    return complex.(sms)

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

    return complex.(sxs), complex.(szs)

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




function sparse_ti_simulation_ideal(N, J,alpha,hx,gamma,dt, steps, m)

    sm =[0 1; 0 0]
    sms=build_lowering_ops(sm, N)
    sxs, szs = build_sparse_spin_ops(sm, N)

    psi = zeros(ComplexF64, 2^N)
    psi[1] = 1.0
    # mneel= div(4^cld(N,2)-1,3)+1
    # psi[mneel] = 1.0
    y = (psi*psi')[:]

    H = build_sparse_hamiltonian(J, alpha, hx, sxs, szs)
    Heff= H
    Heff = complex.(Heff)
    A = spzeros(Float64, 4^N, 4^N)
    Id = sparse(I, 2^N, 2^N)
    Id = complex.(Id)
    A = -1im .*(kron(Id, Heff) - kron(conj.(Heff), Id))

    out_sz=zeros(steps, N )

    for tt= 1: steps
        @show(tt)
        #evaluation 
        rho= reshape(y, 2^N, 2^N)
        for ii = 1:N
            out_sz[tt, ii] = real(tr(rho * szs[ii]))
        end
        y, _ = exponentiate(A,dt, y; krylovdim=m, tol=1e-12)
        # println("Step $tt/$steps - norm(rho) = $(real(tr(reshape(y, 2^N, 2^N))))")
    end

    return out_sz
end



function f_tising(Y, Vij, hx, N)
    xran = 1:N
    yran = (N+1):(2*N)
    zran = (2*N+1):(3*N)

    dY = similar(Y)
    
    for i = xran
        s = 2 * sum(Vij[i,xran] .* Y[zran])
        dY[i] = -Y[N + i] * s
        dY[N + i] = Y[i] * s - 2 * Y[2*N + i] * hx
        dY[2*N + i] = 2 * Y[N + i] * hx
    end
    return dY
end

function RK4(f,Y, Vij, hx, N, dt)
    dt2=dt/2
    k1=f(Y, Vij, hx, N)
    k2=f(Y.+dt2.*k1, Vij, hx, N,)
    k3=f(Y.+dt2.*k2, Vij, hx, N)
    k4=f(Y.+dt.*k3, Vij, hx, N)
    Y+= (dt/6).*(k1 .+ 2*k2 .+ 2*k3 .+ k4)
    return Y

end


function ti_simulation_RK4(f,Y, Vij, hx, N, dt, nt)
    Yres=zeros(Float64, nt,3*N)
    for tt=1:nt
        Y=RK4(f,Y, Vij, hx, N, dt)
        Yres[tt, :]= Y
    end
    return Yres
end


function main()
    Plots.closeall()  
    N =  8# number of spins0
    m=20 # dimension of Krylov space
    J = 1 # defines energy/time units
    alpha = 1# interaction range
    hx = 2 # transverse field
    gamma= 0.0
    dt = 0.1 # time step for plotting
    steps =101
    tran = 0:dt:((steps-1) * dt)




    Y=zeros(Float64, 3*N)
    zran = (2*N+1):(3*N)
    Y[zran] .= -1.0
    
    V= zeros(Float64, N, N)
    for ii = 1:N
        for jj = 1:N
            if ii != jj
                V[ii, jj] = J / abs(ii - jj)^alpha
            end
        end
    end

    Sz_mf=Matrix(undef, steps, N)
    for tt =1:steps
        @show(tt)
        Sz_mf[tt,:].= Y[2*N+1:end]
        Y=RK4(f_tising, Y, V, hx, N, dt)
    end

    Sz_ed=sparse_ti_simulation_ideal(N, J,alpha,hx,gamma,dt, steps, m)
    p= scatter(tran, sum(Sz_ed, dims=2), label="full magnetisation, ideal", title="Transverse Ising model dynamics", markershape=:circle)
    # xlabel!(L"tJ")
    # ylabel!(L"\sum i \langle \hat \sigma_i^z \rangle")
    
    plot!(tran, sum(Sz_mf, dims=2), labels="MF")
    display(p)


    # display(size(out_RK4))
    # display(size(tran))
    # plot()
    # plot!(tran, out_RK4[:,1], label="x of spin 0, RK4", title="Transverse Ising model dynamics", )
    # plot!(tran, out_RK4[:,N+1], label="y of spin 0, RK4", title="Transverse Ising model dynamics")
    # plot!(tran, out_RK4[:,2*N+1], label="z of spin 0, RK4", title="Transverse Ising model dynamics")




        
    # @time out_sz = sparse_ti_simulation_ideal(N, J, alpha, hx, gamma, dt, steps, m)
    # plot!(tran, out_sz[:,3], label="⟨Sz⟩ of spin 3, ideal", title="Transverse Ising model dynamics", linestyle=:dash)
    # plot!(tran, out_sz[:,4], label="⟨Sz⟩ of spin 4, ideal", title="Transverse Ising model dynamics",linestyle=:dash)






    # display(current())
    # cmap = cgrad(:RdBu)
    #     # default(
    #     #     tickfontsize = 10, 
    #     #     labelfontsize = 12, 
    #     #     fontfamily="times",
    #     #     colorbar_ticks=-1:0.5:1,
    #     #     color = cmap,
    #     #     aspect_ratio=1.2,
    #     #     dpi=200)

    #     h = heatmap(1:N, 0:dt:((steps-1) * dt),  out_sz)
    #     xlims!((0.5, N+0.5))
    #     # xlabel!(L"i")
    #     # ylabel!(L"tJ")

    #     display(h)

    return nothing
end

main()


