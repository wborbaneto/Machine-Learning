% Wilson Borba da Rocha Neto
% Genetic Algorithm done in MatLab (Why not?)
% num_cromossomes, ctype='int'
clc; clear all;close all;

cnum = 16;
csize = 9;
num_winners = cnum/2;
max_iterations = 50;

[cromossomes, cindex] = creationism2(cnum, csize);

i = 0;
fitness = 0;
fit_plot = zeros(max_iterations, cnum);
fit_loop = 1;

while fit_loop && i < max_iterations
    
    % Activation function
    [fitness, c] = fit_function2(cromossomes, csize);
    fitness = normalize(fitness);
    % Winner Selection
    winners = selection(cindex, fitness, num_winners);
    % Crossover
    cromossomes = cuddling(csize, cromossomes, winners, num_winners);
    % Mutation
    cromossomes = mutation(cromossomes, cnum, csize);
    
    if sum(size(unique(cromossomes,'rows'))) == csize+1
        fit_loop = 0;
    end
    i = i+1;
    timeline(i,:,:) = cromossomes;
end

% Plot the generations of the first cromossome
cplot2(timeline)

% Functions
function [cromossomes, cindex] = creationism2(cnum, csize)
    % Create all cromossomes.
    header = [0, 0, 0; 0, 0, 1; 1, 0, 0];
    index_header = randi([1, 3],cnum,1);
    cromossomes = [header(index_header,:), randi([0, 1], cnum, csize-3)];
    cindex = 1:cnum;
end

function [cromossomes, cindex] = creationism(cnum, csize)
    % Create all cromossomes.
    cromossomes = randi([0, 1], cnum, csize);
    cindex = 1:cnum;
end

function [fitness] = fit_function(cromossomes)
    % Calculate the fitness.
    fitness =  bin2dec(int2str(cromossomes)).^2;
end

function [fitness, c] = fit_function2(cromossomes, csize)
    % Calculate the fitness.
    array = recursion(csize-1, [2]);
    c = sum([-1, array].* cromossomes, 2);
    fitness =  curve2(c);
    
end

function [n_array] = normalize(array)
    % Normalize based on max and min possible fitness.
    max_fit = max(array);
    min_fit = min(array);
    n_array = (array - min_fit)/(max_fit - min_fit);
end

function [winner] = roulette(fitness, cindex)
    % Calculate the roulette winner.
    %
    % So sad matlab does not have docstring, so i'll add this comments
    % to reproduce pyhton layout.
    %
    % 1. Weight each fitness so their sum is equal to 1
    % 2. Spin the roulette :D
    %
    % 0|--20%--0.2----60%----0.8--20%--|1
    % if win_index < 0.2        (20%) => 1 wins
    % if win_index >0.2 && <0.8 (60%) => 2 wins
    % if win_index >0.8         (20%) => 3 wins
    %
    
    fit = fitness/sum(fitness);
    % Define where the roulette stops
    win_index = rand(1);
    
    for n = 1:length(fit)
        buf(n) = sum(fit(1:n));
        if buf(n) > win_index
            break
        end
    end
    
    winner = cindex(n);
end

function [winners] = selection(cindex, fitness, num_winners)
    % Winners selection method
    cid = cindex;
    fit = fitness;

    for n = 1:num_winners
        winner = roulette(fit, cid);
        winners(n) = winner;
        fit(cid == winner) = [];
        cid(cid == winner) = [];
    end
end

function [new_generation] = cuddling(csize, cromossomes, winners, num_winners)
    cfather = cromossomes(winners,:);
    bit_mask = logical(randi([0,1],1, csize));
    for n = (0:(num_winners-1)/2).*2+1
        cson(n,:) = (cfather(n,:) & bit_mask) + (cfather(n+1,:) & ~bit_mask);
        cson(n+1,:) = (cfather(n+1,:) & bit_mask) + (cfather(n,:) & ~bit_mask);
    end

    new_generation = [cfather; cson];
end

function [muted_cromossomes] = mutation(cromossomes, cnum, csize)
    cromossomes(rand(cnum,1) < 0.02,randi([1, csize])) = randi([0,1]);
    muted_cromossomes = cromossomes;
end

function [y] = curve2(x)
    y = x.*sin(10*pi.*x) + 1;
end

function cplot2(timeline)
    [d, c, r] = size(timeline);
    
    for n = 1:d
        [y,x] = fit_function2(reshape(timeline(n,1,:),[1,r]), r);
        plot(x,y,'*','LineWidth',10), hold on
        legendList{n,1} = int2str(n);
        txt = int2str(n);
        t = text(x,y+0.1,txt);
        t.FontSize = 12;
    end
    legend(legendList)
    plot(x,y,'rs','LineWidth',10)
    x1 = linspace(-1,2,10000);
    y1 = curve2(x1);
    plot(x1,y1, 'LineWidth',2), hold on
end

function [array] = recursion(value,array)
    L = length(array);
    array(L+1) = array(L)/2;
    
    if L < value - 1
        array = recursion(value, array);
    end
end