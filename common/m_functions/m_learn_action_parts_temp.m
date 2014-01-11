% learn actons
% input part features i.e. instance features of the video
% output is the model


function return_state = m_learn_action_parts_v0(global_config)
return_state = 1;

lap_config = global_config.learn_action_parts;
splits = lap_config.splits;
feature_types = lap_config.feature_types;
M5IL = global_config.learn_action_parts.M5IL;

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
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % prepare the bags-instances data
    % instances: store all the instance data [instance_num,instance_dim], the dim is related with the featuer_types
    % bag_labels: [instance_num,1], the bag label of each instance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    instances_cell = cell(length(do_class_idx),1);
    
    for i_class = 1:length(do_class_idx)
        
        cur_class = do_class_idx(i_class);
        
        fprintf('coding split:%d class:%d \n',cur_splits,cur_class);
        cur_vid_idx = train_set_idx{cur_splits}{cur_class};
        
        cur_class_instaces = cell(length(cur_vid_idx),length(feature_types));
        
        for i_feature = 1:length(feature_types)
            cur_feature = feature_types{i_feature};
            
            for i_vid = 1:length(cur_vid_idx)
                cur_vid = cur_vid_idx(i_vid);
                
                % load the video data
                load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature) );
                load(load_name,'parts_features');
                
                cur_class_instaces(i_vid,i_feature) = parts_features;
                
            end
        end
        
        instances_cell{i_class} = cell2mat(cur_class_instaces);
    end
    
    bag_labels = [];
    for i_cell = 1:length(instances_cell)
        if isempty(instances_cell{i_cell})
            fprintf('WRONG!!! no instance in class:%d\n',do_class_idx(i_cell));
            return_state = 0;
            return;
        end
        
        bag_labels = [bag_labels;i_cell*ones( length( size(instances_cell{i_cell},1) ),1 ) ];
    end
    
    
    instances = cell2mat(instances_cell);
    instance_num = length(bag_labels);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % initial W using kmeans SVM
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    for i_class = 1:length(do_class_idx)
        cur_class = do_class_idx(i_class);
        
        W_init = cell(lap_config.params.rand_init_num,1);
        
        for i_init = 1:lap_config.params.rand_init_num
            
            % do kmeans of instances of pos bag
            instances_posbag_idx = find(bag_labels == i_class);
            instances_negbag_idx = find(bag_labels ~= i_class);
            
            instances_posbag = instances(instances_posbag_idx,:);
            
            [~,cluster_idx] = vl_kmeans(instances_posbag',lap_config.params.channel_num,'algorithm', 'elkan');
            
            % include other instances in neg bag
            cur_label = [];
            cur_data = [];
            cur_data = [cur_data;instances(instances_negbag_idx,:)];
            cur_label = [cur_label;zeros(length(instances_negbag_idx) ,1)];
            
            cur_label = [cur_label;cluster_idx'];
            cur_data = [cur_data;instances_posbag];
            
            % doing liblinear classificaiton using multi-class support vector classification by Crammer and Singer
            
            liblinear_option_str = sprintf( '-s 4 -c %f -B %f -q', lap_config.params.liblinear.C,  lap_config.params.liblinear.bias );
            
            model = train_liblinear( cur_label, cur_data, liblinear_option_str );
            cur_W = model.w';
            
            W_init{i_init} = bsxfun( @minus, cur_W(:,2:end), cur_W(:,1) );
            
            
        end %i_init
        
        
        % add save
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CCCP learning
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        
        % learn attribute models for this class
        
        % convert to binary classification problem;
        bag_labels_binary = (-1)*ones( size(bag_labels) );
        bag_labels_binary(instances_posbag_idx) = 1;
        
        % get initial models;
        
        M5IL.init.wInit = W_init;
        
        % perform M5IL;
        [model, info] = do_m5il_solve( instances, indBag_reduced, bag_labels_binary,lap_config.channel_num, M5IL);
        
        action_parts_model = model.w;
        
        %% save <bases> into <dictFolder>;
        save_name = fullfile(lap_config.path,sprintf('s%02d_c%03d.mat',cur_splits,i_class));
        save( save_name, 'action_parts_model');
        
        clear('model');
        clear('action_parts_model');
        
    end % i_class
    
   
    
end % i_split



end