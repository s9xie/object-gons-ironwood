% learn actons
% input part features i.e. instance features of the video
% output is the model


function return_state = m_learn_action_parts_v0_vc(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx',
% 'train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

lap_config = global_config.learn_action_parts;
splits = lap_config.splits;
feature_types = lap_config.feature_types;
M5IL = global_config.learn_action_parts.M5IL;

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

%vl_setup;

for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
       
    for i_feature = 1:length(feature_types)
        cur_feature = feature_types{i_feature};
        
        fprintf('loding instances split:%d feature:%s\n',cur_splits,cur_feature);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % prepare the bags-instances data
        % instances: [instance_num,instance_dim], all the instance data, the dim is related with the featuer_types
        % instance_bag_labels:[instance_num,1],  the bag labels of each instances
        % bag_class_labels: [instance_num,1], the class labels of each instances
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if global_config.learn_action_parts.reload
            
            instances_cell = cell(length(do_class_idx),1);
            instance_bag_labels = [];
            bag_count = 0;
            bag_class_labels = [];
            for i_class = 1:class_num
                
                cur_class = i_class;             
                cur_img_idx = train_set_idx{cur_splits}{cur_class};                
                cur_class_instaces = cell(length(cur_img_idx),length(feature_types));
                                              
                for i_img = 1:length(cur_img_idx)
                    cur_vid = cur_img_idx(i_img);
                   
                    % load the video data
                    load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature) );
                    load(load_name,'parts_features');
                 
                    cur_class_instaces{i_img,i_feature} = (parts_features);
                 
                    bag_count = bag_count+1;
                    %instance_bag_labels = [instance_bag_labels;bag_count*ones( size(parts_features,1),1 ) ];
                    instance_bag_labels = [instance_bag_labels;bag_count*ones( size(parts_features,2),1 ) ];
                    %WARNING: NOT SURE ABOUT THAT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                    bag_class_labels = [bag_class_labels;i_class];
                    
                end
                
                instances_cell{i_class} = cell2mat(cur_class_instaces);
            end
            
            instances = cell2mat(instances_cell);
            clear instances_cell;
            instance_num = length(instance_bag_labels);
            
            instance_class_labels = bag_class_labels(instance_bag_labels);
%             save_name = fullfile(lap_config.path,sprintf('instances_data_s%02d_%s.mat',cur_splits,cur_feature) );
%             save(save_name,'instances','instance_bag_labels','bag_class_labels','instance_class_labels','-v7.3')
        else
            
            load_name = fullfile(lap_config.path,sprintf('instances_data_s%02d_%s.mat',cur_splits,cur_feature) );
            if exist(load_name,'file')
                fprintf('loading instances data ... \n');
                load(load_name,'instances','instance_bag_labels','bag_class_labels','instance_class_labels');
            else
                fprintf('WRONG!!! instances data not exist!\n');
                return_state = 0;
                return;
            end
            
        end
             
        for i_class = 1:length(do_class_idx)
            cur_class = do_class_idx(i_class);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % initial W using kmeans SVM
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            W_init = cell(M5IL.init.rand_init_num,1);
            
            
            for i_init = 1:M5IL.init.rand_init_num
                
                fprintf('feature:%s initing class:%3d init:%2d ... ',cur_feature,cur_class,i_init);
               
                % do kmeans of instances of pos bag
                instances_posbag_idx = find(instance_class_labels == cur_class);
                instances_negbag_idx = find(instance_class_labels ~= cur_class);
                
                instances_posbag = instances(instances_posbag_idx,:);
                
                [~,cluster_idx] = vl_kmeans((instances_posbag)',M5IL.channel_num,'algorithm', 'elkan','numRepetitions',3);
                
                % include other instances in neg bag
                
                cur_label = zeros(length(instance_class_labels),1);
                cur_label(instances_posbag_idx) = cluster_idx';
                
                % doing liblinear classificaiton using multi-class support vector classification by Crammer and Singer
                
                liblinear_option_str = sprintf( '-s 4 -c %f -B %f -q', M5IL.init.liblinear.C,  M5IL.init.liblinear.bias );
                model = liblinear_train( cur_label, sparse(instances), liblinear_option_str );
                
                cur_W = model.w' ;
                neg_label_idx = (model.Label == 0);
                pos_label_idx = (model.Label ~= 0);
                cur_W = model.w';
                cur_W_pos = cur_W(:,pos_label_idx);
                cur_W_neg = cur_W(:,neg_label_idx);
                
                cur_W = bsxfun( @minus, cur_W_pos, cur_W_neg );
                
                if global_config.learn_action_parts.init.using_rand
                    cur_W = cur_W + 0.1*randn(size(cur_W));
                end
                
                W_init{i_init} = bsxfun( @minus, cur_W(:,2:end), cur_W(:,1) );
                
               
                
            end %i_init
                        
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % CCCP learning
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % convert to binary classification problem;
            bag_labels_binary = (-1)*ones( size(bag_class_labels) );
            bag_labels_binary( find(bag_class_labels == cur_class)) = 1;
            
            % get initial models;
            
            M5IL.init.wInit = W_init;
            
            % perform M5IL;
            [model, info] = do_m5il_solve( [(instances),ones( size(instances,1),1 )], instance_bag_labels, bag_labels_binary,M5IL.channel_num, M5IL);
            
            % save the model
            action_parts_model = model.w;            
            save_name = fullfile(lap_config.path,sprintf('s%02d_c%03d_%s-%d.mat',cur_splits,cur_class,cur_feature,global_config.learn_action_parts.struct_M5IL.graph.node_num));
            save( save_name, 'action_parts_model','info');

            clear('model');
            clear('action_parts_model');
            
            
        end % i_class
        
    end % i_feature
    
end % i_split



end