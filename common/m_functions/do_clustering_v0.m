
% do_clustering(cur_class,cur_class_vids(i_vid),clustering_config,feature_path)
function  do_clustering(cur_class,cur_vid,clustering_config,feature_path)

fprintf('clustering c:%02d v:%03d ',cur_class,cur_vid);

cluster_results_path = clustering_config.path;
cluster_params = clustering_config.params;

const_c_cluster_num = 1e-4;
const_dis_v_eps = 1e-2;

NN = cluster_params.NN +1; %neighbor number of each node
spatial_dis_thread = clustering_config.params.spatial_dis_thread; % max dis between
exp_beta = clustering_config.params.exp_beta; % weight = exp(exp_beta * dis);
co_exist_len = clustering_config.params.co_exist_len;
min_cluster_size = clustering_config.params.min_cluster_size;
max_cluster_num = clustering_config.params.max_cluster_num;


%first load the dt_feature
load_name = fullfile(feature_path, sprintf('c%03d_v%03d_location',cur_class,cur_vid));
load(load_name,'location');

load_name = fullfile(feature_path, sprintf('c%03d_v%03d_trajectory',cur_class,cur_vid));
load(load_name,'trajectory');

dt_len = size(trajectory,2)/2;
N = size(trajectory,1);
%the final cluster id
cluster_id = zeros(N,1);

%--------------------------------------------
% compute all the edges
%--------------------------------------------
tic;
t_begin = location(:,1);
x = trajectory(:,1:2:end);
y = trajectory(:,2:2:end);
clear trajectory;

v_x = diff(x,1,2);
v_y = diff(y,1,2);

edges = cell(N,1);

tic;
for i = 1:N
    
    % find time co_exist
    t_delay = t_begin - t_begin(i);
    t_delay_max = (dt_len - co_exist_len);
    
    % get the co_exist part
    co_exist_id = find( abs(t_delay) <= t_delay_max);
    
    co_exist_delay = t_delay(co_exist_id);
    co_exist_x = x(co_exist_id,:);
    co_exist_y = y(co_exist_id,:);
    
    
    neighbor_id = [];
    neighbor_weight = [];
    
    
    for i_delay = - t_delay_max:1:t_delay_max
        
        cur_delay_select_id = find(co_exist_delay == i_delay);
        
        if (length(cur_delay_select_id) > 0)
            % compute the coexist_part
            
            if (i_delay < 0)
                cur_co_exist_x = co_exist_x(cur_delay_select_id,end+1-dt_len-i_delay:end);
                cur_co_exist_y = co_exist_y(cur_delay_select_id,end+1-dt_len-i_delay:end);
                dis = (cur_co_exist_x - repmat(x(i,1:dt_len+i_delay),length(cur_delay_select_id),1)).^2 + (cur_co_exist_y - repmat(y(i,1:dt_len+i_delay),length(cur_delay_select_id),1)).^2;
            else
                cur_co_exist_x = co_exist_x(cur_delay_select_id,1:dt_len-i_delay);
                cur_co_exist_y = co_exist_y(cur_delay_select_id,1:dt_len-i_delay);
                dis = (cur_co_exist_x - repmat(x(i,end+1-dt_len+i_delay:end),length(cur_delay_select_id),1)).^2 + (cur_co_exist_y - repmat(y(i,end+1-dt_len+i_delay:end),length(cur_delay_select_id),1)).^2;
            end
            
            dis_max = max(dis,[],2);
            dis_near_id = find(dis_max < spatial_dis_thread^2);
            
            if length(dis_near_id) >0
                
                dis_near = dis_max(dis_near_id);
                if length(dis_near_id) > NN
                    [~,sort_id] = sort(dis_near,'ascend');
                    dis_near_id = dis_near_id(sort_id(1:NN));
                end
                
                cur_neighbor_id = co_exist_id(cur_delay_select_id(dis_near_id));
                cur_neighbor_dis_spatial = sqrt(dis_max(dis_near_id));
                
                if (i_delay < 0)
                    cur_co_exist_v_x = v_x(cur_neighbor_id,end+1-(dt_len-1)-i_delay:end);
                    cur_co_exist_v_y = v_y(cur_neighbor_id,end+1-(dt_len-1)-i_delay:end);
                    dis_v = (cur_co_exist_v_x - repmat(v_x(i,1:(dt_len-1)+i_delay),length(cur_neighbor_id),1)).^2 + (cur_co_exist_v_y - repmat(v_y(i,1:(dt_len-1)+i_delay),length(cur_neighbor_id),1)).^2;
                else
                    cur_co_exist_v_x = v_x(cur_neighbor_id,1:(dt_len-1)-i_delay);
                    cur_co_exist_v_y = v_y(cur_neighbor_id,1:(dt_len-1)-i_delay);
                    dis_v = (cur_co_exist_v_x - repmat(v_x(i,end+1-(dt_len-1)+i_delay:end),length(cur_neighbor_id),1)).^2 + (cur_co_exist_v_y - repmat(v_y(i,end+1-(dt_len-1)+i_delay:end),length(cur_neighbor_id),1)).^2;
                end
                
                cur_neighbor_dis_v = mean(sqrt(dis_v),2) + const_dis_v_eps;
                
                cur_neighbor_weight = exp(exp_beta * (cur_neighbor_dis_spatial.*cur_neighbor_dis_v));
                
                
                neighbor_id = [neighbor_id;cur_neighbor_id];
                neighbor_weight = [neighbor_weight;cur_neighbor_weight];
            end    %  if lenght(select_id) >0
            
        end % if(length(cur_delay_select_id) > 0)
        
        
        
    end % (i_delay = - t_delay_max:1:t_delay_max)
    
    neighbor_weight( find(neighbor_id ==i ) ) = 1e9;
    
    if length(neighbor_id) > 1
        [~,sort_id] = sort(neighbor_weight,'descend');
        if ( length(neighbor_id) > NN )
            sort_id = sort_id(1:NN);
        end
        
        edges{i} = [i*ones(length(sort_id)-1,1),neighbor_id(sort_id(2:end)),neighbor_weight(sort_id(2:end))];
        %             edges{i} = [neighbor_id(sort_id(2:end)),neighbor_weight(sort_id(2:end))];
        %affinity_matrix_up(i,neighbor_id) = neighbor_weight;
    end
end % for i = 1:N-1

toc;
tic;

edges = cell2mat(edges);
affinity_matrix_up = sparse(edges(:,1),edges(:,2),edges(:,3),N,N);
affinity_matrix = max(affinity_matrix_up , affinity_matrix_up');


%--------------------------------------------
% compute the groups of the graph
%--------------------------------------------

edge_links = cell(N,1);
for i = 1:N
    [~, edge_links{i},~] = find(affinity_matrix(i,:));
end

node_group_id = cmpt_graph_groups(edge_links);
group_num = max(node_group_id);
groups = cell(group_num,1);
groups_size = zeros(group_num,1);

for i = 1:group_num
    groups{i} = find(node_group_id == i);
    groups_size(i) = length(groups{i});
end

toc;
tic;

%--------------------------------------------
% do ganc
%--------------------------------------------

% save('temp_ganc.mat','groups','affinity_matrix','group_num','affinity_matrix_up','x','y','t_begin');

% load('temp_ganc.mat','groups','affinity_matrix','group_num','affinity_matrix_up','x','y','t_begin');

cluster_count = 0;

for i = 1:group_num
    cur_group_node_num = length(groups{i});
    if (cur_group_node_num > 2*min_cluster_size)  % a big group, at least get 2 clusters
        %get group affinity matrix
        cur_affinity_matrix = affinity_matrix(groups{i},groups{i});
        
        % compute the eigenvalue to get the cluster num ------------------
        %         tic;
        %         eigen_vals = svds(cur_affinity_matrix);
        %         toc;
        %         cs_eigen_vals = cumsum(eigen_vals);
        %         sc_val = eigen_vals(2:end)./cs_eigen_vals(1:end-1)+const_c_cluster_num*[2:(length(cs_eigen_vals))]';
        %         [~, cluster_num ] = min(sc_val);
        
        % ----------------------------------------------------------------
        cluster_num = round(cur_group_node_num/(2*min_cluster_size)); % the parameter should be updated
        cluster_num = min(cluster_num,max_cluster_num);
        
        
        
        % ganc -----------------------
        % input: [affinity matrix,edges_num,cluster_num], affinity matrix,only up
        % part,and the cluster_num is the cluster computed by the eigenval
        % output: cluster_id
        %-----------------------------
        cur_affinity_matrix_up = affinity_matrix_up(groups{i},groups{i});
        [~,~,cur_edges_weight] = find(cur_affinity_matrix_up);
        
        cur_edges_num = length(cur_edges_weight);
        
        % save('debug_ganc_data.mat','cur_affinity_matrix_up','cur_edges_num','cluster_num');
        
        cur_cluster_id = ganc_mex(cur_affinity_matrix_up,cur_edges_num,cluster_num);
        cur_cluster_num = max(cur_cluster_id);
        for cur_i_cluster = 1:cur_cluster_num
            
            cur_cluster_size = sum(cur_cluster_id == cur_i_cluster);
            if cur_cluster_size > min_cluster_size
                
                cluster_count = cluster_count +1;
                cluster_id(groups{i}(find(cur_cluster_id == cur_i_cluster)) ) = cluster_count;
                
            end
            
        end
        
    elseif cur_group_node_num > min_cluster_size
        
        cluster_count = cluster_count +1;
        cluster_id(groups{i}) = cluster_count;
        
    end % cur_group_node_num ends
    
end

% check the cluster number

if cluster_count == 0
    fprintf('all small groups!\n');
    
    max_group_size = max(groups_size);
    % get the bigest groups
    
    for i = 1:group_num
        if groups_size(i) > 0.5*max_group_size
            
            cluster_count = cluster_count +1;
            cluster_id(groups{i}) = cluster_count;
        end
        
    end
    
end


fprintf(' cluster_num:%5d ',max(cluster_id));
toc;
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
    %     temp_c = input('');
end

close all;

end