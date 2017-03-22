function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    
    true_positives = 0;
    false_positives = 0;
    false_negatives = 0;
    
    % if pval < epsilon then it is considered to be an anomaly
    predictions = (pval < epsilon);
                    
    % y = 1 corresponds to an anomalous example, 
    % and y = 0 corresponds to a normal example
    
    for i=1:size(predictions)
        if(yval(i) == 1 && predictions(i) == 1)
            true_positives = true_positives + 1;
        elseif (yval(i) == 0 && predictions(i) == 1)
            false_positives = false_positives + 1;
        elseif(yval(i) == 1 && predictions(i) == 0)
            false_negatives = false_negatives + 1;
        end
    end
    
    prec = true_positives / (true_positives + false_positives);
    rec= true_positives / (true_positives + false_negatives);

    F1 = 2 * prec * rec / (prec + rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
