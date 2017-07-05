function [mmse, mapm] = bac_compute_estimator(y, H, option, results, misc)


mmse = struct();

[M, N] = size(H);

AfterBurn = option.nburn+1;


% MMSE
mmse.x      = mean(results.x_all(:, AfterBurn:end), 2);

if strcmp('bac1', misc.type(1:4))
    mmse.sigma2 = mean(results.sigma2_all(:, AfterBurn:end));
    mmse.mu     = mean(results.mu_all(:, AfterBurn:end));
    
elseif strcmp('bac2', misc.type(1:4))
    mmse.beta = mean(results.beta_all(:, AfterBurn:end));
else
    error('results type not recognize');
end





% MAPm
mapm = struct();
if strcmp('bac1', misc.type(1:4))
    
    if option.sample_X && ~option.sample_mu && ~option.sample_sigma2
    
        beta = 2 * results.mu_all * N * results.sigma2_all;
        
        min = inf;
        for it = 1:size(results.x_all, 2)
            x =  results.x_all(:, it);
            buf = norm(y - H * x)^2 + beta * norm(x, inf);
            
            if buf < min
                buf = min;
                mapm.x = x;
            end
        end
        
        
        
        
        
    elseif option.sample_X && option.sample_mu && option.sample_sigma2
        
        min = inf;
        for it = 1:size(results.x_all, 2)
            x =  results.x_all(:, it);
            buf = (option.a_sigma2 + M /2) * log( option.b_sigma2 + norm(y - H * x)^2 ) ...
                + (option.a_mu + N) * log( option.b_mu + N * norm(x, inf) );
            
            if buf < min
                buf = min;
                mapm.x = x;
            end
        end
        
    else
        error('Mapm not Implemented');
    end
    
    
    
    
    
    
    
elseif strcmp('bac2', misc.type(1:4))
    
    if option.sample_X && ~option.sample_beta
        
        beta = results.beta_all;
        min = inf;
        for it = 1:size(results.x_all, 2)
            x =  results.x_all(:, it);
            buf = norm(y - H * x)^2 + beta * norm(x, inf);
            
            if buf < min
                buf = min;
                mapm.x = x;
            end
        end
        
        
        
    elseif option.sample_X && option.sample_beta
        
        mu = results.mu;
        min = inf;
        for it = 1:size(results.x_all, 2)
            x =  results.x_all(:, it);
            buf = (mu * N) * norm(x, inf) ...
              + (option.a_beta + M/2) * log( option.b_beta + norm(y - H * x)^2 );
            
            if buf < min
                buf = min;
                mapm.x = x;
            end
        end        
        
    else
        error('Mapm not Implemented');
    end
    
    
    
    
    
    
    
else
    error('results type not recognize');
end