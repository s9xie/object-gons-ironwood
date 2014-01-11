% v1: using mex to compute affinity matrix and group id
% v2: add hirarchy cluster
% do_clustering(cur_class,cur_class_vids(i_vid),clustering_config,feature_path)
function  do_clustering_v2(cur_class,cur_vid,clustering_config,feature_path)



cluster_results_path = clustering_config.path;
cluster_params = clustering_config.params;

const_c_cluster_num = 1e-4;
const_dis_v_eps = 1e-2;

NN = cluster_params.NN +1; %neighbor number of each node
spatial_dis_thread = clustering_config.params.spatial_dis_thread; % max dis between
exp_beta = clustering_config.params.exp_beta; % weight = exp(exp_beta * dis);
co_exist_len = clustering_config.params.co_exist_len;
min_cluster_size = clustering_config.params.min_cluster_size;


target_cluster_num = clustering_config.params.target_cluster_num;

min_cluster_num = clustering_config.params.min_cluster_num;


%first load the dt_feature
load_name = fullfile(feature_path, sprintf('c%03d_v%03d_location',cur_class,cur_vid));
load(load_name,'location');

load_name = fullfile(feature_path, sprintf('c%03d_v%03d_trajectory',cur_class,cur_vid));
load(load_name,'trajectory');

load_name = fullfile(feature_path, sprintf('c%03d_v%03d_info',cur_class,cur_vid));
load(load_name,'vid_info');

dt_len = size(trajectory,2)/2;
N = size(trajectory,1);


%--------------------------------------------
% compute all the edges
%--------------------------------------------


t_begin = location(:,1);
x = trajectory(:,1:2:end);
y = trajectory(:,2:2:end);
clear trajectory;

% tr_t = t_begin;
% tr_x = x';
% tr_y = y';
% save('debug_computing_affinity_matrix.mat','tr_t','tr_x','tr_y','NN','spatial_dis_thread','co_exist_len','exp_beta');

[affinity_matrix,group_idx]= do_computing_affinity_matrix_mex(t_begin,x',y',NN,spatial_dis_thread,co_exist_len,exp_beta);

group_num = max(group_idx);
groups = cell(group_num,1);
groups_size = zeros(group_num,1);

for i = 1:group_num
    groups{i} = find(group_idx == i);
    groups_size(i) = length(groups{i});
end


%--------------------------------------------
% do ganc
%--------------------------------------------

cluster_candidate_count = 0;
clusters_candidate = {};

for i = 1:group_num
    cur_group_node_num = length(groups{i});
    
    cur_group_cluster_num_target = round(cur_group_node_num/N * target_cluster_num); % the parameter should be updated
    cur_group_cluster_num_target = min(cur_group_cluster_num_target,round(cur_group_node_num/(min_cluster_size*2)));
    
    if (cur_group_cluster_num_target >= 2)  % a big group, at least get 2 clusters
        %get group affinity matrix
        cur_affinity_matrix = affinity_matrix(groups{i},groups{i});
      
        
        
        % ganc -----------------------
        % input: [affinity matrix,edges_num,cluster_num], affinity matrix,only up
        % part,and the cluster_num is the cluster computed by the eigenval
        % output: cluster_id
        %-----------------------------
        [~,~,cur_edges_weight] = find(cur_affinity_matrix);
        
        cur_edges_num = length(cur_edges_weight);
        
        % save('debug_ganc_data.mat','cur_affinity_matrix_up','cur_edges_num','cluster_num');
        
        hirarchy_ratio = cluster_params.hirarchy_ratio;
        for i_level = 1:length(hirarchy_ratio)
            cur_level_cluster_num_target = round(cur_group_cluster_num_target*(hirarchy_ratio(i_level)));
            
            if (cur_level_cluster_num_target < 2)
                break;
            end
            
            cur_cluster_id = ganc_mex(cur_affinity_matrix,cur_edges_num,cur_level_cluster_num_target);
            cur_level_cluster_num = max(cur_cluster_id);
            
            for cur_i_cluster = 1:cur_level_cluster_num
                
                cluster_candidate_count = cluster_candidate_count +1;
                clusters_candidate{cluster_candidate_count} = groups{i}(find(cur_cluster_id == cur_i_cluster));
                
            end
            
        end
        
    else
        
        cluster_candidate_count = cluster_candidate_count +1;
        clusters_candidate{cluster_candidate_count} = groups{i};
        
    end % cur_group_node_num ends
    
end

% check the cluster number

clusters = {};
cluster_count = 0;
small_cluster_candidates_idx = [];
small_cluster_candidates_size = [];

for i_cluster = 1:cluster_candidate_count
    if length(clusters_candidate{i_cluster}) >= min_cluster_size
        cluster_count = cluster_count+1;
        clusters{cluster_count} = clusters_candidate{i_cluster};  
    else
        small_cluster_candidates_idx = [small_cluster_candidates_idx;i_cluster];
        small_cluster_candidates_size = [small_cluster_candidates_size;length(clusters_candidate{i_cluster})];
    end    
end


% when all the cluster is small, choose small clusters bigger than min_cluster_size* 0.3, until get min_cluster_num 
if cluster_count < min_cluster_num
    %fprintf('get all small groups!\n');    
    [sorted_size,sorted_idx] = sort(small_cluster_candidates_size,'descend');
    
    for i_small_cluster = 1:length(sorted_size)
        if (cluster_count < min_cluster_num ) %&& sorted_size(i_small_cluster) > min_cluster_size* 0.1)
            
            cluster_count = cluster_count+1;
            clusters{cluster_count} = clusters_candidate{small_cluster_candidates_idx(sorted_idx(i_small_cluster))};
            
        end
    end
end



% remove the same clusters
for i = 1:length(clusters)
    if isempty(clusters{i})
        continue;
    end
    for j = i+1 : length(clusters)
        if length(clusters{j}) == length(clusters{i})
            if sum(clusters{j} == clusters{i}) == length(clusters{i})
                clusters{j} = [];
            end
        end       
    end
end

cluster_id = {};
cluster_count = 0;

for i = 1:length(clusters)
    if ~isempty(clusters{i})
        cluster_count = cluster_count +1;
        cluster_id{cluster_count} = clusters{i};
    end
end


fprintf('clustering c:%03d v:%03d  cluster_num:%5d \n',cur_class,cur_vid,cluster_count);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save_name = fullfile(clustering_config.path,sprintf('clusters_c%03d_v%03d.mat',cur_class,cur_vid));

save(save_name,'cluster_id');

% plot_cluster(cluster_id,x,y,t_begin,dt_len,vid_info,cur_class,cur_vid,cluster_results_path);

end



function node_group_id = cmpt_graph_groups(edges)

N = length(edges);
node_group_id = zeros(N,1);

group_count = 1;

while(sum(node_group_id==0)>0)
    
    i = find(node_group_id==0,1,'first');
    node_group_id(i) = group_count;
    
    adjacency_set = i;
    
    while(size(adjacency_set) > 0)
        
        pop_node = adjacency_set(1);
        if ( length(adjacency_set) >1 )
            adjacency_set = adjacency_set(2:end);
        else
            adjacency_set = [];
        end
        
        if (length(edges{pop_node}) > 0)
            cur_edges = edges{pop_node};
            pushback_nodes = cur_edges(find( node_group_id(cur_edges) ==0 ));
            
            if (length(pushback_nodes) > 0)
                node_group_id(pushback_nodes) = group_count;
                adjacency_set = [adjacency_set,pushback_nodes];
            end
        end
        
    end
    
    group_count = group_count+1;
end


end

function plot_cluster(cluster_id,x,y,t_begin,dt_len,vid_info,iClass,iFile,cluster_results_path)

img_save_path = sprintf('%s\\cluster_results_show',cluster_results_path);
if ~exist(img_save_path)
    mkdir(img_save_path);
end

h = figure;

cluster_num = max(cluster_id);
t_begin = t_begin - dt_len +2;

% vid_info = [vid_width;vid_height;vid_len;fea_dim];

x = vid_info(1) - x;
y = vid_info(2) - y;


cluster_sizes = zeros(cluster_num,1);
for i_cluster = 1:cluster_num
    cluster_sizes(i_cluster) = sum(cluster_id == i_cluster);
end

[~,cluster_id_sorted] = sort(cluster_sizes,'descend');


for i_cluster = 1:cluster_num
    
    fprintf('i_cluster %d/cluster_num %d\n',i_cluster,cluster_num);
    
    cur_id = find(cluster_id == cluster_id_sorted(i_cluster));
    cur_x = x(cur_id,:);
    cur_y = y(cur_id,:);
    temp_t = t_begin(cur_id);
    cur_t = temp_t;
    for temp_i = 1:dt_len-1
        cur_t = [cur_t,temp_t+temp_i];
    end
    N = size(cur_x,1);
    
    t_min = min(min(cur_t));
    t_max = max(max(cur_t));
    
    t_mean = [t_min:t_max]';
    x_mean = zeros(length(t_mean),1);
    y_mean = zeros(length(t_mean),1);
    
    ind = 1;
    for i_t = t_min:t_max
        
        select_id = find(cur_t == i_t);
        x_mean(ind) = mean(cur_x(select_id));
        y_mean(ind) = mean(cur_y(select_id));
        ind = ind +1;
    end
    
    show_num = 30;
    if N > show_num
        select_id = randperm(N,show_num);
        
        cur_x = cur_x(select_id,:);
        cur_y = cur_y(select_id,:);
        cur_t = cur_t(select_id,:);
        
    end
    
    N = size(cur_x,1);
    
    rand_color = [rand rand rand];
    
    %     if (i_cluster == 4 || i_cluster == 11 ||i_cluster == 18)
    %         plot3(t_mean,x_mean,y_mean,'color',rand_color,'LineWidth',5);
    %         hold on;
    %     end
    
    plot3(t_mean,x_mean,y_mean,'color',rand_color,'LineWidth',5);
    hold on;
    
    %     for i = 1:N
    %         plot3(cur_t(i,:),cur_x(i,:),cur_y(i,:),'color',rand_color,'LineWidth',1);
    %         hold on;
    %     end
    
    grid on;
    axis([0,vid_info(3),0,vid_info(1),0,vid_info(2)]);
    xlabel('t');
    ylabel('x');
    zlabel('y');
    
    fig_path = sprintf('%s\\c%02d\\v%03d',img_save_path,iClass,iFile);
    
    if ~exist(fig_path)
        mkdir(fig_path);
    end
    
    fig_name = sprintf('%s\\img%03d',fig_path,i_cluster);
    
    saveas(h,fig_name,'bmp');
        temp_c = input('');
end

close all;

end