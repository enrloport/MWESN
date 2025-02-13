function full_log(params,params_esn,mwE;extra=Dict())
    p,pe=params,params_esn
    to_log = Dict(
        "Total time"        => p[:total_time]
        ,"Train time"       => p[:train_time]
        ,"Test time"        => p[:test_time]
        ,"Layers"           => p[:layers]
        , "Sigmoids"        => pe[:sgmds]
        , "Alphas"          => pe[:alpha]
        , "Densities"       => pe[:density]
        , "W_in_densities"  => pe[:Win_dens]
        , "Rhos"            => pe[:rho]
        , "Sigmas"          => pe[:sigma]
        , "W_scalings"      => pe[:W_scaling]
        , "reservoirs"      => sum([x[1] for x in p[:layers]])
        , "nodes" => sum( [ sum(l) for l in p[:layers] ] )
	    , "alpha min" => minimum( vcat( pe[:alpha]...) )
	    , "alpha max" => maximum( vcat( pe[:alpha]...) )
	    , "density min" => minimum( vcat( pe[:density]...) )
	    , "density max" => maximum( vcat( pe[:density]...) )
        , "rho" => pe[:rho][1][1]
        , "sigma" => pe[:sigma][1][1]
    )
    err_dict = Dict("Error_step_"*string(s) => mwE.error[s] for s in params[:steps] )
    merge!(to_log,err_dict,extra)
    cls_nms = string.(p[:classes])
    if p[:wb]
        if p[:confusion_matrix]
            for stp in p[:steps]
                to_log["conf_mat_"*string(stp)] = Wandb.wandb.plot.confusion_matrix(
                    y_true = p[:test_labels][stp], preds = [x[1] for x in mwE.Y[stp]], class_names = cls_nms
                )
            end
        end
        Wandb.log(p[:lg], to_log )
    else
        display(to_log)
        if p[:confusion_matrix]
            for stp in p[:steps]
                display(confusion_matrix(cls_nms, p[:test_labels][stp], [x[1] for x in mwE.Y[stp]]) )
            end
        end
    end
end