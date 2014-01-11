% learn actons
% input part features i.e. instance features of the video
% output is the model

% v1 add pair feature
function return_state = m_learn_action_parts_v1(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx',
% 'train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

lap_config = global_config.learn_action_parts;
splits = lap_config.splits;
feature_types = lap_config.feature_types;
struct_M5IL = global_config.learn_action_parts.struct_M5IL;

struct_M5IL.graph.edge_num = struct_M5IL.graph.node_num*(struct_M5IL.graph.node_num-1)/2;
load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_pair.mat',1,1,1) );
load(load_name,'parts_pair_features');
struct_M5IL.graph.edge_dim = size(parts_pair_features,2);

% get which classes to learn
switch class(global_config.classification.class_idx)
    case 'double'
        do_class_idx = global_config.classification.class_idx;
        
    case 'char'
        do_class_idx = [1:class_num];
        
    otherwise
        fprintf('WRONG!!! wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end

vl_setup;

for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
    
    for i_feature = 1:length(feature_types)
        cur_feature = feature_types{i_feature};
        
        fprintf('loding instances split:%d feature:%s\n',cur_splits,cur_feature);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % prepare the bags-instances data
        % instances: store all the instance data [instance_num,instance_dim], the dim is related with the featuer_types
        % bag_labels: [instance_num,1], the bag label of each instance
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if global_config.learn_action_parts.reload
            
            instances_cell = cell(length(do_class_idx),1);
            instance_bag_labels = [];
            bag_count = 0;
            bag_class_labels = [];
            bag_paths = {};
            for i_class = 1:length(do_class_idx)
                
                cur_class = do_class_idx(i_class);
                
                %                 fprintf('loding instances split:%d class:%d feature:%s\n',cur_splits,cur_class,cur_feature);
                cur_vid_idx = train_set_idx{cur_splits}{cur_class};
                
                for i_vid = 1:length(cur_vid_idx)
                    cur_vid = cur_vid_idx(i_vid);
                    bag_count = bag_count+1;
                    
                    load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature) );
                    % load the video data ---------------------------------
                    %
                    %                     load(load_name,'parts_features');
                    %                       cur_class_instaces{i_vid,i_feature} = parts_features;
                    %                       instance_bag_labels = [instance_bag_labels;bag_count*ones( size(parts_features,1),1 ) ];
                    % -----------------------------------------------------
                    bag_class_labels = [bag_class_labels;i_class];
                    pair_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_pair.mat',cur_splits,cur_class,cur_vid) );
                    cur_bag_path = cell(1,2);
                    cur_bag_path{1} = load_name;
                    cur_bag_path{2} = pair_name;
                    bag_paths{bag_count,1} = cur_bag_path;
                    
                end
                
                
            end
            
            
            
            %             save_name = fullfile(lap_config.path,sprintf('instances_data_s%02d_%s.mat',cur_splits,cur_feature) );
            %             save(save_name,'instances','instance_bag_labels','bag_class_labels','instance_class_labels','-v7.3')
        else
            
            load_name = fullfile(lap_config.path,sprintf('instances_data_s%02d_%s.mat',cur_splits,cur_feature) );
            if exist(load_name,'file')
                fprintf('loading instances data ... \n');
                load(load_name,'instances','instance_bag_labels','bag_class_labels','instance_class_labels','bag_paths');
            else
                fprintf('WRONG!!! instances data not exist!\n');
                return_state = 0;
                return;
            end
            
        end
        
        
        W_init_all_class = cell(length(do_class_idx),1);
        
        % initial the W of node parts -------------------------------------
        %         for i_class = 1:length(do_class_idx)
        %             cur_class = do_class_idx(i_class);
        %
        %
        %             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %             % initial W using kmeans SVM
        %             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        %             fprintf('feature:%s initing class:%3d ... \n',cur_feature,cur_class);
        %
        %             % do kmeans of instances of pos bag
        %             instances_posbag_idx = find(instance_class_labels == i_class);
        %             instances_negbag_idx = find(instance_class_labels ~= i_class);
        %
        %             instances_posbag = instances(instances_posbag_idx,:);
        %
        %             [~,cluster_idx] = vl_kmeans((instances_posbag)',struct_M5IL.channel_num * struct_M5IL.graph.node_num ,'algorithm', 'elkan','numRepetitions',3);
        %
        %
        %             fprintf('cluster_size: ')
        %             for i_cluster = 1:max(cluster_idx)
        %                 fprintf('% 5d ',length(find(cluster_idx == i_cluster)));
        %             end
        %             fprintf('\n');
        %             % include other instances in neg bag
        %
        %             cur_label = zeros(length(instance_class_labels),1);
        %             cur_label(instances_posbag_idx) = cluster_idx';
        %             liblinear_option_str = sprintf( '-s 4 -c %f -B %f -q', struct_M5IL.init.liblinear.C,  struct_M5IL.init.liblinear.bias );
        %
        %             model = liblinear_train( cur_label, sparse(instances), liblinear_option_str );
        %
        %             neg_label_idx = find(model.Label == 0);
        %             pos_label_idx = find(model.Label ~= 0);
        %             cur_W = model.w';
        %             cur_W_pos = cur_W(:,pos_label_idx);
        %             cur_W_neg = cur_W(:,neg_label_idx);
        %
        %             cur_W = bsxfun( @minus, cur_W_pos, cur_W_neg );
        %
        %             struct_M5IL.graph.node_dim = size(cur_W,1);
        %
        %
        %             W_init = cell(struct_M5IL.init.rand_init_num,1);
        %             for i_init = 1:struct_M5IL.init.rand_init_num
        %                 node_rand_idx = randperm(struct_M5IL.graph.node_num * struct_M5IL.channel_num);
        %
        %                 W_init{i_init} = [];
        %
        %                 ind = 0;
        %                 for i_channel = 1:struct_M5IL.channel_num
        %
        %                     cur_channel_W = [];
        %                     for i_node = 1:struct_M5IL.graph.node_num
        %                         ind = ind +1;
        %                         cur_channel_W = [cur_channel_W; cur_W(:,ind)];
        %                     end
        %
        %                     cur_channel_W = [cur_channel_W;zeros(struct_M5IL.graph.edge_num*struct_M5IL.graph.edge_dim,1)];
        %
        %                     W_init{i_init} = [W_init{i_init},cur_channel_W];
        %                 end
        %             end
        %
        %             save('W_init.mat','W_init');
        %             W_init_all_class{i_class} = W_init;
        %         end % i_class learn W_init ends
        
        % -------------------------------------------------------------------------
        
        for i_class = 1:length(do_class_idx)
            cur_class = do_class_idx(i_class);
            
            
            fprintf('-----learning model class %3d ------\n',cur_class);
            % not debug --------------------
            % W_init = W_init_all_class{cur_class};
            
            % load for debug ----------------------------------------------------------
            load('W_init_rand.mat','W_init');
            load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,1,1,cur_feature) );
            load(load_name,'parts_features');
            struct_M5IL.graph.node_dim = size(parts_features,2)+1;
            
            % load for debug ----------------------------------------------------------
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % CCCP learning
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            bag_pos_idx = find(bag_class_labels == cur_class);
            bag_neg_idx = find(bag_class_labels ~= cur_class);
            
            % every row is [parts_features_path, parts_pair_feature_path ]
            bag_paths_pos = bag_paths(bag_pos_idx,:);
            bag_paths_neg = bag_paths(bag_neg_idx,:);
            
            % perform M5IL;
            
%             W_size = size(W_init{1});
%             W_init{1} = 0.01*randn(W_size);
            
            [model, info] = do_struct_m5il_solve_v0(bag_paths_pos, bag_paths_neg,  W_init,struct_M5IL);
            
            
            action_parts_model = model.W;
            
            % check the cccp results --------------------------------------
            %             scores = [(instances),ones( size(instances,1),1 )]*action_parts_model;
            %             bag_scores = zeros(max(instance_bag_labels),2);
            %             for i_bag = 1:max(instance_bag_labels)
            %                 cur_bag_score = scores(find(instance_bag_labels == i_bag),:);
            %
            %                 bag_scores(i_bag,1) = max(max(cur_bag_score));
            %                 %bag_scores(i_bag,2) = ceil(find(cur_bag_score == bag_scores(i_bag,1))/size(cur_bag_score,1));
            %             end
            %             % min pos bag score and max neg bag score
            %             pos_bag_idx = find(bag_labels_binary == 1);
            %             neg_bag_idx = find(bag_labels_binary == -1);
            %             min_pos_bag_score = min(bag_scores(pos_bag_idx,1));
            %             max_neg_bag_score = max(bag_scores(neg_bag_idx,1));
            %             fprintf('min_pos_bag_score:%f max_neg_bag_score:%f\n\n',min_pos_bag_score,max_neg_bag_score);
            %             save_name = fullfile(lap_config.path,sprintf('cccp_results_s%02d_c%03d_%s.mat',cur_splits,i_class,cur_feature));
            %             save( save_name, 'bag_scores','min_pos_bag_score','max_neg_bag_score');
            % ------------------------------------------------------------
            
            save_name = fullfile(lap_config.path,sprintf('s%02d_c%03d_%s.mat',cur_splits,i_class,cur_feature));
            save( save_name, 'action_parts_model','info');
            
            clear('model');
            clear('action_parts_model');
            
            
        end % i_class
        
    end % i_feature
    
end % i_split



end

%%
function cur_W = do_init_kmean_svm(cur_class,cur_init,bag_labels,instances,lap_config)
% do kmeans of instances of pos bag
fprintf('initing class:%3d init:%2d ... \n',cur_class,cur_init);
instances_posbag_idx = find(bag_labels == cur_class);
instances_negbag_idx = find(bag_labels ~= cur_class);

instances_posbag = instances(instances_posbag_idx,:);

[~,cluster_idx] = vl_kmeans(full(instances_posbag)',lap_config.channel_num,'algorithm', 'elkan');

cluster_size = zeros(lap_config.channel_num,1);
for i_cluster = 1:lap_config.channel_num
    cluster_size(i_cluster) =  length(find(cluster_idx == i_cluster));
end
fprintf('cluster_size %4d %4d %4d\n',cluster_size(1),cluster_size(2),cluster_size(3));

% include other instances in neg bag

cur_label = zeros(size(instances,1),1);
cur_label(instances_posbag_idx) = cluster_idx';

% doing liblinear classificaiton using multi-class support vector classification by Crammer and Singer

if cur_class == 1  % make sure the first data is zero label
    
    instances = full(instances);
    
    temp_instance = instances(end,:);
    instances(end,:) = instances(1,:);
    instances(1,:) = temp_instance;
    
    temp_label = cur_label(end);
    cur_label(end) = cur_label(1);
    cur_label(1) = temp_label;
    
    instances = sparse(instances);
end

liblinear_option_str = sprintf( '-s 4 -c %f -B %f -q', lap_config.params.liblinear.C,  lap_config.params.liblinear.bias );

model = liblinear_train( cur_label, instances, liblinear_option_str );
cur_W = model.w';
end