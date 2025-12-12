#!/usr/bin/env julia
# RWHEC calibration using certifiable SDP solver

using LinearAlgebra
using Rotations
using DelimitedFiles
using SparseArrays
using JuMP

solver_path = joinpath(@__DIR__, "..", "third_party", "certifiable-rwhe-calibration")
include(joinpath(solver_path, "src", "calibration", "robot_world_costs.jl"))
include(joinpath(solver_path, "src", "rotation_sdp_solver_jump.jl"))

function quat_trans_to_poses(csv_path::String)
    data = readdlm(csv_path, ',', Float64)
    n = size(data, 1)
    poses = zeros(n, 4, 4)
    poses[:, 4, 4] .= 1
    for i in 1:n
        qw, qx, qy, qz = data[i, 1:4]
        poses[i, 1:3, 1:3] = Matrix(QuatRotation(qw, qx, qy, qz))
        poses[i, 1:3, 4] = data[i, 5:7]
    end
    return poses
end

function find_calibration_files(data_dir::String)
    files = Dict{Tuple{Int, Int}, Tuple{String, String, Int}}()
    for file in readdir(data_dir)
        m = match(r"tag_(\d+)_cam_(\d+)_A\.csv", file)
        if m !== nothing
            tag_id = parse(Int, m.captures[1])
            cam_idx = parse(Int, m.captures[2])
            a_path = joinpath(data_dir, file)
            b_path = joinpath(data_dir, "tag_$(tag_id)_cam_$(cam_idx)_B.csv")
            if isfile(b_path)
                files[(tag_id, cam_idx)] = (a_path, b_path, countlines(a_path))
            end
        end
    end
    return files
end

function verify_certification(Z, model; tol=1e-4)
    primal = objective_value(model)
    dual = dual_objective_value(model)
    gap = abs(primal - dual) / (1.0 + abs(primal))
    
    eigenvalues = eigvals(Hermitian(Matrix(Z)))
    sorted_eigs = sort(real.(eigenvalues), rev=true)
    
    println("\nCertification:")
    println("  Primal: $(round(primal, digits=2))")
    println("  Dual:   $(round(dual, digits=2))")
    println("  Gap:    $(round(gap, sigdigits=3))")
    
    certified = gap < tol
    println("  Status: $(certified ? "CERTIFIED" : "NOT CERTIFIED")")
    
    return certified, gap, sorted_eigs
end

function main()
    data_dir = joinpath(@__DIR__, "..", "output", "julia_data")
    output_dir = joinpath(@__DIR__, "..", "output", "final")
    mkpath(output_dir)

    excluded_tags = Set([9, 15, 20])
    min_poses = 50
    
    all_files = find_calibration_files(data_dir)
    valid_pairs = Dict{Tuple{Int, Int}, Tuple{String, String, Int}}()
    all_tags, all_cams = Set{Int}(), Set{Int}()

    for ((tag_id, cam_idx), (a_path, b_path, n)) in all_files
        if tag_id in excluded_tags
            continue
        end
        if n >= min_poses
            valid_pairs[(tag_id, cam_idx)] = (a_path, b_path, n)
            push!(all_tags, tag_id)
            push!(all_cams, cam_idx)
        end
    end

    sorted_tags = sort(collect(all_tags))
    sorted_cams = sort(collect(all_cams))
    n_cams = length(sorted_cams)
    n_tags = length(sorted_tags)

    println("Cameras: $sorted_cams")
    println("Tags: $sorted_tags")
    println("Valid pairs: $(length(valid_pairs))")

    tag_to_idx = Dict(t => j for (j, t) in enumerate(sorted_tags))
    cam_to_idx = Dict(c => i for (i, c) in enumerate(sorted_cams))

    # Build cost matrix
    Q = get_empty_sparse_cost_matrix(n_cams, n_tags, false)
    half_langevin_k = 6.0
    trans_weight = 0.5 / 0.05^2
    total_poses = 0

    for ((tag_id, cam_idx), (a_path, b_path, n)) in valid_pairs
        eye_idx = cam_to_idx[cam_idx]
        base_idx = tag_to_idx[tag_id]
        
        A = quat_trans_to_poses(a_path)
        B = quat_trans_to_poses(b_path)
        
        Q_temp = sparse_robot_world_transformation_cost(
            A, B, 
            half_langevin_k .* ones(n), 
            trans_weight .* ones(n)
        )
        Q += get_ith_eye_jth_base_cost(eye_idx, base_idx, n_cams, n_tags, Q_temp, false)
        total_poses += n
    end

    println("Total poses: $total_poses")
    println("Solving SDP...")
    
    Z, model = solve_sdp_dual_using_concat_schur(Q, n_cams + n_tags, true, true)
    certified, gap, eigs = verify_certification(Z, model)
    solution = extract_solution_from_dual_schur(Z, Q)

    # Extract and save results
    println("\nResults:")
    Xs = Dict{Int, Matrix{Float64}}()
    
    for (i, cam_idx) in enumerate(sorted_cams)
        t = solution[3*(i-1)+1 : 3*i]
        R = nearest_rotation(reshape(solution[9*(i-1) + 3*(n_cams+n_tags) + 1 : 9*i + 3*(n_cams+n_tags)], 3, 3))
        X = zeros(4, 4); X[1:3, 1:3] = R; X[1:3, 4] = t; X[4, 4] = 1.0
        Xs[cam_idx] = X
        
        q = QuatRotation(R)
        name = cam_idx == 0 ? "left" : "right"
        println("  X_$name: t=$(round.(t, digits=3))")
        
        X_csv = vcat(Rotations.params(q), t)
        writedlm(joinpath(output_dir, "$(name)_X.csv"), X_csv', ',')
        writedlm(joinpath(output_dir, "$(name)_X_matrix.csv"), X, ',')
    end

    for (j, tag_id) in enumerate(sorted_tags)
        idx = n_cams + j
        t = solution[3*(idx-1)+1 : 3*idx]
        R = nearest_rotation(reshape(solution[9*(idx-1) + 3*(n_cams+n_tags) + 1 : 9*idx + 3*(n_cams+n_tags)], 3, 3))
        Y = zeros(4, 4); Y[1:3, 1:3] = R; Y[1:3, 4] = t; Y[4, 4] = 1.0
        
        q = QuatRotation(R)
        Y_csv = vcat(Rotations.params(q), t)
        writedlm(joinpath(output_dir, "Y_tag_$(tag_id).csv"), Y_csv', ',')
    end

    # Save certification
    open(joinpath(output_dir, "certification.txt"), "w") do f
        println(f, "duality_gap: $gap")
        println(f, "certified: $certified")
        println(f, "top_eigenvalues: $(eigs[1:min(5,length(eigs))])")
    end

    if haskey(Xs, 0) && haskey(Xs, 1)
        baseline = norm(Xs[0][1:3, 4] - Xs[1][1:3, 4])
        println("\nBaseline: $(round(baseline*100, digits=2)) cm")
    end
    
    println("Saved to $output_dir")
end

main()