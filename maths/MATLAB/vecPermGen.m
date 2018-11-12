% given integer total and min, returns all permutations of integers above
% min that add up to total
function vectors = vecPermGen(total, min)
    if total < min*2
        vectors = {total};
    else
        vectors = {};
        for h1 = min:total-min;
            vecs = vecPermGen(total-h1, min);
            sz = size(vectors,1);
            for i = 1:size(vecs,1)
                vectors{sz+i,1} = [h1 vecs{i}]; %#ok<*AGROW>
            end
        end
        sz = size(vectors,1);
        vectors{sz+1,1} = total;
    end
end