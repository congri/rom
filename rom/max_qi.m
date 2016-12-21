function [Xmax] = max_qi(qi, startValue)
%Maximize distribution qi

    function [lg_qi_neg, d_lg_qi_neg] = log_qi_neg(X)
        [lg_qi, d_lg_qi] = qi(X);
        lg_qi_neg = -lg_qi;
        d_lg_qi_neg = -d_lg_qi;
    end

log_qi_neg_handle = @(X) log_qi_neg(X);
options = optimoptions('fminunc', 'GradObj', 'on', 'Algorithm', 'quasi-newton', 'Display', 'off');
Xmax = fminunc(log_qi_neg_handle, startValue, options);


end

