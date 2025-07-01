using Plots
using JuMP
using Ipopt

# Paramètres du modèle de Monod
s_in = 6.0       
μ1_max = 1.7     
μ2_max = 1.8     
K1 = 0.3         
K2 = 0.6         
D_max = 1.5      

# Fonctions de Monod
μ1(s) = μ1_max * s / (K1 + s)  
μ2(s) = μ2_max * s / (K2 + s)  
Δ(s) = μ1(s) - μ2(s)          

# Calcul de s_bar (valeur qui maximise Δ(s))
s_values = range(0, s_in, length=1000)
s_bar = s_values[argmax(Δ.(s_values))]

# Paramètres de simulation
N = 10000        
t0, tf = 0.0, 6.0   
Δt = (tf - t0) / N  

# Création du modèle d'optimisation
model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 5))
set_optimizer_attribute(model, "tol", 1e-8)
set_optimizer_attribute(model, "constr_viol_tol", 1e-6)
set_optimizer_attribute(model, "max_iter", 1000)

# Définition des variables
@variables(model, begin
    0 ≤ s[1:N+1] ≤ s_in  
    0 ≤ x1[1:N+1]        
    0 ≤ x2[1:N+1]        
    0 ≤ D[1:N+1] ≤ D_max 
end)

# Conditions initiales
@constraints(model, begin
    s[1] == 2.0  
    x1[1] == 2.0 
    x2[1] == 2.0 
end)

# Dynamique du modèle (Crank-Nicolson)
@NLconstraints(model, begin
    [i = 1:N], s[i+1] == s[i] + 0.5 * Δt * ((D[i] * (s_in - s[i]) - μ1(s[i]) * x1[i] - μ2(s[i]) * x2[i]) + (D[i+1] * (s_in - s[i+1]) - μ1(s[i+1]) * x1[i+1] - μ2(s[i+1]) * x2[i+1]))
    [i = 1:N], x1[i+1] == x1[i] + 0.5 * Δt * ((μ1(s[i]) - D[i]) * x1[i] + (μ1(s[i+1]) - D[i+1]) * x1[i+1])
    [i = 1:N], x2[i+1] == x2[i] + 0.5 * Δt * ((μ2(s[i]) - D[i]) * x2[i] + (μ2(s[i+1]) - D[i+1]) * x2[i+1])
end)

# Objectif : Maximiser x1(tf) / x2(tf) (ajout d'une régularisation pour éviter division par 0)
@NLobjective(model, Max, x1[N+1] / (x2[N+1] + 1e-6))

# Résolution du modèle
println("Résolution du modèle de Monod...")
optimize!(model)

# Récupération des résultats
s_opt, x1_opt, x2_opt, D_opt = value.(s), value.(x1), value.(x2), value.(D)

# Calcul du contrôle singulier D_s(t)
μ1_bar, μ2_bar = μ1(s_bar), μ2(s_bar)
D_s = [ (μ1_bar * x1_opt[i] + μ2_bar * x2_opt[i]) / (x1_opt[i] + x2_opt[i]) for i in 1:N+1 ]

# Temps pour les graphiques
t = range(t0, tf, length=N+1)

# --- GRAPHIQUES AVEC STYLE AMÉLIORÉ ---

# 1️⃣ Évolution des concentrations x1(t) et x2(t)
p1 = plot(t, x1_opt, label="Espèce \$x_1(t)\$", linewidth=8, color=:blue)
plot!(p1, t, x2_opt, label="Espèce \$x_2(t)\$", linewidth=8, color=:red)
title!(p1, "Évolution des concentrations")

# 2️⃣ Évolution du substrat s(t) avec s_bar
p2 = plot(t, s_opt, label="Substrat \$s(t)\$", xlabel="Temps", ylabel="Concentration", linewidth=8, color=:purple)
hline!(p2, [s_bar], label="\$\\bar{s}\$ = $(round(s_bar, digits=3))", color=:red, linestyle=:dot, linewidth=5)
title!(p2, "Évolution du substrat")

# 3️⃣ Évolution des contrôles D(t) et D_s(t)
p3 = plot(t, D_opt, label="Contrôle \$D(t)\$", xlabel="Temps", ylabel="Taux de dilution", linewidth=8, color=:green)
plot!(p3, t, D_s, label="Contrôle singulier \$D_s(t)\$", linewidth=8, linestyle=:dash, color=:orange)
title!(p3, "Évolution des contrôles")

# Amélioration de la légende pour tous les graphes
# Appliquer cette configuration à tous les graphiques
for p in [p1, p2, p3]
    plot!(p, legend=:right, legend_position=:center, 
          legendfontsize=12, legend_background_color=:white, 
          legend_titlefontsize=16)
end


# Affichage final des graphiques en colonne
plot(p1, p2, p3, layout=(3, 1), size=(900, 1000))

# Sauvegarde du graphique
savefig("monod_model_optimized.pdf")
