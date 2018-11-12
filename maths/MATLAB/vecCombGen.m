% given integer total and min, returns all combinations of integers above
% min that add up to total
function vectors = vecCombGen(total, min)
    if total < min
        vectors = {};
    elseif total < min*2
        vectors = {total};
    else
        vectors = {};
        for h1 = min:total-min;
            vecs = vecCombGen(total-h1, min);
            sz = size(vectors,1);
            for i = 1:size(vecs,1)
                vectors{sz+i,1} = [h1 vecs{i}]; %#ok<*AGROW>
            end
            min = min+1;
        end
        sz = size(vectors,1);
        vectors{sz+1,1} = total;
    end
end