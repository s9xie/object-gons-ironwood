% learn actons
% input part features i.e. instance features of the video
% output is the model

% v1 add pair feature
% v2 suppor do specific classes and add rand
function return_state = m_learn_action_parts_v2(global_config)


clear parts_features_all;
clear parts_pair_features_all;

clear bag_idx_pos; % the bag index of pos
clear bag_idx_neg; % the bag index of neg

clear parts_bag_idx; % the parts_features_all index of each bags
clear parts_pair_bag_idx;


global parts_features_all; 
global parts_pair_features_all; % all the parts_pair features

global bag_idx_pos; % the bag index of pos
global bag_idx_neg; % the bag index of neg

global parts_bag_idx; % the parts_features_all index of each bags
global parts_pair_bag_idx; % the parts_pair_features_all index of each bags
global bag_class_labels;

return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx',
% 'train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

lap_config = global_config.learn_action_parts;
splits = lap_config.splits;
feature_types = lap_config.feature_types;
struct_M5IL = global_config.learn_action_parts.struct_M5IL;


% get which classes to learn
switch class(global_config.learn_action_parts.class_idx)
    case 'double'
        do_class_idx = global_config.learn_action_parts.class_idx;
        
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
        
        struct_M5IL.graph.edge_num = struct_M5IL.graph.node_num*(struct_M5IL.graph.node_num-1)/2;
        load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_pair.mat',1,1,1) );
        load(load_name,'parts_pair_features');
        struct_M5IL.graph.edge_dim = size(parts_pair_features,2);
        
        load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,1,1,cur_feature) );
        load(load_name,'parts_features');
        struct_M5IL.graph.node_dim = size(parts_features,2)+1;
        
        struct_M5IL.graph.psi_dim = struct_M5IL.graph.node_dim * struct_M5IL.graph.node_num + struct_M5IL.graph.edge_dim * struct_M5IL.graph.edge_num;
        
        if 1
%             if global_config.learn_action_parts.reload
            
            bag_count = 0;
            parts_count = 0;
            parts_pair_count = 0;            
            bag_class_labels = [];

            % get the size of all parts features
            % must load all the class
            for i_class = 1:class_num
                
                cur_class = i_class;
                cur_vid_idx = train_set_idx{cur_splits}{cur_class};
                
                for i_vid = 1:length(cur_vid_idx)
                   cur_vid = cur_vid_idx(i_vid);
                   
                   load_name = fullfile(global_config.clustering.path,sprintf('clusters_c%03d_v%03d.mat',cur_class,cur_vid));
                   load(load_name,'cluster_id');
                   
                   switch class(cluster_id)
                       case 'double'
                           cluster_num = max(cluster_id);
                       case 'cell'
                           cluster_num = length(cluster_id);
                       otherwise
                           fprintf('WRONG!!! type of cluster_id wrong\n');
                           return;
                   end
                   
                   bag_count = bag_count+1;
                   bag_class_labels = [bag_class_labels;i_class];
                   
                   parts_count_old = parts_count +1;
                   parts_count = parts_count + cluster_num;
                   parts_bag_idx{bag_count,1} = parts_count_old:parts_count;
                   
                   parts_pair_count_old = parts_pair_count +1;
                   parts_pair_count = parts_pair_count + cluster_num*(cluster_num+1)/2;
                   parts_pair_bag_idx{bag_count,1} = parts_pair_count_old:parts_pair_count;
                   
                end
             
            end
            
            parts_features_all = zeros(parts_count,struct_M5IL.graph.node_dim );
            parts_pair_features_all = zeros(parts_pair_count,struct_M5IL.graph.edge_dim );
            
            bag_count = 0; 
            fprintf('begin loading....\n');
            tic; 
            for i_class = 1:class_num
                
                cur_class = i_class;                
                cur_vid_idx = train_set_idx{cur_splits}{cur_class};
                
                for i_vid = 1:length(cur_vid_idx)
                    cur_vid = cur_vid_idx(i_vid);
                    load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature) );
                    load(load_name,'parts_features');
                    
                    pair_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_pair.mat',cur_splits,cur_class,cur_vid) );
                    load(pair_name,'parts_pair_features');
                    
                   
                    bag_count = bag_count+1;
                    parts_features_all( parts_bag_idx{bag_count},:) = [parts_features, ones( size(parts_features,1) ,1 )];
                    parts_pair_features_all( parts_pair_bag_idx{bag_count} ,:) = parts_pair_features;
                    
                end
                
            end
            toc;
            fprintf('end loading....\n');
        else
            
            
        end
      
        % initial the W of node parts -------------------------------------
        for i_class = 1:length(do_class_idx)
            cur_class = do_class_idx(i_class);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % initial W using kmeans SVM
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            fprintf('feature:%s initing class:%3d ... \n',cur_feature,cur_class);
            
            bag_idx_pos = find(bag_class_labels == cur_class);
            bag_idx_neg = find(bag_class_labels ~= cur_class);
            
            % do kmeans of instances of pos bag
            instances_posbag_idx = [];
            for i_bag = 1:length(bag_idx_pos)
                instances_posbag_idx = [instances_posbag_idx; parts_bag_idx{ bag_idx_pos(i_bag) }'];
            end
            
            instances_negbag_idx = [];
            for i_bag = 1:length(bag_idx_neg)
                instances_negbag_idx = [instances_negbag_idx; parts_bag_idx{ bag_idx_neg(i_bag) }'];
            end
            
            % do kmeans on all the pos intances
            instances_posbag = parts_features_all(instances_posbag_idx,1:end-1);            
            [~,cluster_idx] = vl_kmeans((instances_posbag)',struct_M5IL.channel_num * struct_M5IL.graph.node_num ,'algorithm', 'elkan','numRepetitions',3);
            fprintf('cluster_size: ')
            for i_cluster = 1:max(cluster_idx)
                fprintf('% 5d ',length(find(cluster_idx == i_cluster)));
            end
            fprintf('\n');
            
            % include other instances in neg bag to do liblinear
            % multi-class classification
            
            instance_bag_labels = [instances_posbag_idx;instances_negbag_idx];
            
            cur_label = zeros(length(instance_bag_labels),1);
            cur_label(instances_posbag_idx) = cluster_idx';
            liblinear_option_str = sprintf( '-s 4 -c %f -q', struct_M5IL.init.liblinear.C);
            model = liblinear_train( cur_label, sparse(parts_features_all), liblinear_option_str );
            
            neg_label_idx = (model.Label == 0);
            pos_label_idx = (model.Label ~= 0);
            cur_W = model.w';
            cur_W_pos = cur_W(:,pos_label_idx);
            cur_W_neg = cur_W(:,neg_label_idx);
            
            cur_W = bsxfun( @minus, cur_W_pos, cur_W_neg );
            
            % get the initial W of struct_M5IL using cur_W
            W_init = cell(struct_M5IL.init.rand_init_num,1);
            for i_init = 1:struct_M5IL.init.rand_init_num
                node_rand_idx = randperm(struct_M5IL.graph.node_num * struct_M5IL.channel_num);
                
                W_init{i_init} = [];
                
                ind = 0;
                for i_channel = 1:struct_M5IL.channel_num
                    
                    cur_channel_W = [];
                    for i_node = 1:struct_M5IL.graph.node_num
                        ind = ind +1;
                        cur_channel_W = [cur_channel_W; cur_W(:,node_rand_idx(ind))];
                    end
                    
                    cur_channel_W = [cur_channel_W;zeros(struct_M5IL.graph.edge_num*struct_M5IL.graph.edge_dim,1)];
                    
                    if global_config.learn_action_parts.init.using_rand
                        cur_channel_W = cur_channel_W + 0.1*randn(size(cur_channel_W));
                    end
                    W_init{i_init} = [W_init{i_init},cur_channel_W];
                end
            end
            

            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % CCCP learning
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [model, info] = do_struct_m5il_solve_v1(W_init,struct_M5IL);
            
            % save model
            action_parts_model = model.W;    
            save_name = fullfile(lap_config.path,sprintf('s%02d_c%03d_%s-%d.mat',cur_splits,cur_class,cur_feature,global_config.learn_action_parts.struct_M5IL.graph.node_num));
            save( save_name, 'action_parts_model','info');
            
            clear('model');
            clear('action_parts_model');
            
            
        end % i_class
        
    end % i_feature
    
end % i_split



end
