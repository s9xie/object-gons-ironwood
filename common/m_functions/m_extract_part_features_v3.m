% extract part data from trajectory features, grouping method are: VOI or
% cluster
% output: c%03d_v%03d_[feature_type].mat
% c%03d_v%03d_location.mat
%

% v1: pooling on the addaptive VOI
% v2: add parts_pair_feature,but all zeros
% v3: real parts_pair_features
% v4: suppot not syn pair
function return_state = m_extract_part_features_v3(global_config)
return_state = 1;

% load dataset info
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'
load(global_config.read_dataset_info.file_name);

extract_part_features_config = global_config.extract_part_features;
splits = extract_part_features_config.splits;
feature_types = extract_part_features_config.feature_types;

do_sets = global_config.extract_part_features.class_idx;

do_vids = cell(length(splits),class_num);

switch class(do_sets)
    case 'cell'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(do_sets)
                do_vids{cur_splits,do_sets{i}(1)} = [do_vids{cur_splits,do_sets{i}(1)};do_sets{i}(2)];
            end
            
        end
    case 'double'
        
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(do_sets)
                do_vids{cur_splits,do_sets(i)} = [train_set_idx{cur_splits}{do_sets(i)};test_set_idx{cur_splits}{do_sets(i)}];
            end
        end
        
    case 'char'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:class_num
                do_vids{cur_splits,i} = [train_set_idx{cur_splits}{i};test_set_idx{cur_splits}{i}];
            end
        end
        
    otherwise
        fprintf('wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end


for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
    
    for i_class = 1:class_num
        
        cur_class = i_class;
        if isempty(do_vids{cur_splits,i_class})
            continue;
        end
        
        for i_feature = 1:length(feature_types)
            cur_feature = feature_types{i_feature};
            
            fprintf('extract part feature split:%2d class:%3d  feature:%s\n',cur_splits,cur_class,cur_feature);
            % all data
            cur_vid_idx = do_vids{cur_splits,i_class};
            
            switch extract_part_features_config.params.method
                case 'cluster'
                    
                    if global_config.num_core > 1
                        parfor i_vid = 1:length(cur_vid_idx)
                            do_pooling_cluster_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),extract_part_features_config,...
                                global_config.coding.path,global_config.extract_features.path,global_config.clustering.path);
                        end
                    else
                        for i_vid = 1:length(cur_vid_idx)
                            do_pooling_cluster_max(cur_splits,cur_feature,cur_class,cur_vid_idx(i_vid),extract_part_features_config,...
                                global_config.coding.path,global_config.extract_features.path,global_config.clustering.path);
                        end
                    end
                    
                otherwise
                    fprintf('wrong type of extract_part_features_config.params.method %s!\n',extract_part_features_config.params.method);
                    return_state = 0;
                    return;
            end
            
        end  % ifeature_type
    end % iclass ends
    
end % i_split

end
%%

function do_pooling_cluster_max(cur_splits,cur_feature,cur_class,cur_vid,extract_part_features_config, codes_path,feature_path,clusters_path)

EPS = 1e-6;

fprintf('extract part feature split:%2d class:%3d vid:%3d feature:%s\n',cur_splits,cur_class,cur_vid,cur_feature);

load_name = fullfile(clusters_path,sprintf('clusters_c%03d_v%03d.mat',cur_class,cur_vid));
load(load_name,'cluster_id');

switch class(cluster_id)
    case 'double'
        cluster_num = max(cluster_id);
    case 'cell'
        cluster_num = length(cluster_id);
    otherwise
        fprintf('WRONG!!! type of cluster_id wrong\n');
end

if strcmp(cur_feature,'pair')
    
    pair_params = extract_part_features_config.pair_params;
    
    % load trajectory feature
    load_name = fullfile(feature_path,sprintf('c%03d_v%03d_trajectory.mat',cur_class,cur_vid));
    load(load_name,'trajectory');
    load_name = fullfile(feature_path,sprintf('c%03d_v%03d_location.mat',cur_class,cur_vid));
    load(load_name,'location');
    load_name = fullfile(feature_path,sprintf('c%03d_v%03d_info.mat',cur_class,cur_vid));
    
    % [width, height, len, dt_number]
    load(load_name,'vid_info');
    
    % compute pair feature ---------------------------------------
    
    loc_t_all = cell(cluster_num,1);
    loc_x_all = cell(cluster_num,1);
    loc_y_all = cell(cluster_num,1);
    vel_x_all = cell(cluster_num,1);
    vel_y_all = cell(cluster_num,1);
    
    for i_cluster = 1:cluster_num
        
        % compute cluster pair feature      
        switch class(cluster_id)
            case 'double'
                select_id_i = find(cluster_id == i_cluster);
            case 'cell'
                select_id_i = cluster_id{i_cluster};
            otherwise
                fprintf('WRONG!!! type of cluster_id wrong\n');
        end
        
        
        cur_location = location(select_id_i,:);
        
        x_low = min(cur_location(:,2));
        x_high = max(cur_location(:,2));
        
        y_low = min(cur_location(:,3));
        y_high = max(cur_location(:,3));
        
        t_low = min(cur_location(:,1));
        t_high = max(cur_location(:,1));
        
        VOI_select_id = (location(:,2) >= x_low)&(location(:,2) <= x_high)&...
            (location(:,3) >= y_low)&(location(:,3) <= y_high )&...
            (location(:,1) >= t_low)&(location(:,1) <= t_high);
        
        trajectory_i = trajectory(VOI_select_id,:);
        x_i = trajectory_i(:,1:2:end)/vid_info(1);
        y_i = trajectory_i(:,2:2:end)/vid_info(2);
        
        %         v_x_i = diff(x_i,1,2);
        %         v_y_i = diff(y_i,1,2);
        v_x_i = x_i(:,end) - x_i(:,1);
        v_y_i = y_i(:,end) - y_i(:,1);      
        location_t_i = location(VOI_select_id,1);
        location_x_i = location(VOI_select_id,2)/vid_info(1);
        location_y_i = location(VOI_select_id,3)/vid_info(2);
        
        
        cur_loc_t = [];
        cur_loc_x = [];
        cur_loc_y = [];
        cur_vel_x = [];
        cur_vel_y = [];
        
        cell_t = min(location_t_i):5:max(location_t_i);
        cell_t = [cell_t,max(location_t_i)];
        
        for t = 1:length(cell_t) -1
            
            cur_idx = location_t_i >= cell_t(t)& location_t_i <= cell_t(t+1);
            if sum(cur_idx) == 0
                continue;   
            end
            cur_loc_t = [cur_loc_t;mean(location_t_i(cur_idx))];
            cur_loc_x = [cur_loc_x;mean(location_x_i(cur_idx))];
            cur_loc_y = [cur_loc_y;mean(location_y_i(cur_idx))];
            cur_vel_x = [cur_vel_x;mean(v_x_i(cur_idx))];
            cur_vel_y = [cur_vel_y;mean(v_y_i(cur_idx))];
            
            
%             if sum(isnan(cur_loc_t))
%                 a = 1;
%             end
            
        end
        
       
        
        loc_t_all{i_cluster} = cur_loc_t;
        loc_x_all{i_cluster} = cur_loc_x;
        loc_y_all{i_cluster} = cur_loc_y;
        vel_x_all{i_cluster} = cur_vel_x;
        vel_y_all{i_cluster} = cur_vel_y;
                
    end

    edge_dim = ( length(pair_params.location_mu))^2+ (length(pair_params.motion_mu))^2+ (length(pair_params.time_mu));
    parts_pair_features = zeros(cluster_num*(cluster_num+1)/2,edge_dim);
    
    ind = 0;
    for i_cluster = 1:cluster_num
        
        i_num = size(loc_t_all{i_cluster},1);
        location_t_i = loc_t_all{i_cluster};
        location_x_i = loc_x_all{i_cluster};
        location_y_i = loc_y_all{i_cluster};
        v_x_i = vel_x_all{i_cluster};
        v_y_i = vel_y_all{i_cluster};
        
        for j_cluster = i_cluster:cluster_num
            % compute cluster pair feature
            ind = ind +1;
            
            j_num = size(loc_t_all{j_cluster},1);
            location_t_j = loc_t_all{j_cluster};
            location_x_j = loc_x_all{j_cluster};
            location_y_j = loc_y_all{j_cluster};
            v_x_j = vel_x_all{j_cluster};
            v_y_j = vel_y_all{j_cluster};
            
            
            mat_loc_t = repmat(location_t_i,1,j_num) - repmat(location_t_j',i_num,1);
            mat_loc_t = abs(mat_loc_t);
            
            mat_loc_x = repmat(location_x_i,1,j_num) - repmat(location_x_j',i_num,1);
            mat_loc_x = abs(mat_loc_x);
            
            mat_loc_y = repmat(location_y_i,1,j_num) - repmat(location_y_j',i_num,1);
            mat_loc_y = abs(mat_loc_y);
            
            
            mat_mot_x = repmat(v_x_i,1,j_num) - repmat(v_x_j',i_num,1);
            mat_mot_x = abs(mat_mot_x);
            
            mat_mot_y = repmat(v_y_i,1,j_num) - repmat(v_y_j',i_num,1);
            mat_mot_y = abs(mat_mot_y);
            
                     
            % pair_params.location_mu = [0,0.1,0.2,0.4,0.8];
            % pair_params.location_var = [1,1,1,1,1];
            % pair_params.motion_mu = [0,0.1,0.2,0.4,0.8];
            % pair_params.motion_mu = [1,1,1,1,1];
            % pair_params.time_mu = [0,5,10,20,40,80];
            % pair_params.time_mu = [1,1,1,1,1,1];
            
            pair_loc_t_feat = zeros(length(pair_params.time_mu),1);
            
            for i = 1:length(pair_params.time_mu)                
                pair_loc_t_feat(i) = mean(mean(exp(-(mat_loc_t - pair_params.time_mu(i)).^2/ pair_params.time_var(i))));
            end
            if sum(pair_loc_t_feat) > EPS
             pair_loc_t_feat = pair_loc_t_feat/sum(pair_loc_t_feat);
            end
            
            pair_loc_x_feat = zeros(length(pair_params.location_mu),1);
            for i = 1:length(pair_params.location_mu)
                pair_loc_x_feat(i) = mean( mean( exp(-(mat_loc_x - pair_params.location_mu(i)).^2/pair_params.location_var(i)))) ;
            end
            
            if sum(pair_loc_x_feat) > EPS
                pair_loc_x_feat = pair_loc_x_feat/sum(pair_loc_x_feat);
            end
            
            pair_loc_y_feat = zeros(length(pair_params.location_mu),1);
            for i = 1:length(pair_params.location_mu)
                pair_loc_y_feat(i) =  mean( mean( exp(-(mat_loc_y - pair_params.location_mu(i)).^2/pair_params.location_var(i)))) ;
            end
            
            if sum(pair_loc_y_feat) > EPS
            pair_loc_y_feat = pair_loc_y_feat/sum(pair_loc_y_feat);
            end
            pair_loc_xy_feat = pair_loc_x_feat*pair_loc_y_feat';
                                
            
            pair_mot_x_feat = zeros(length(pair_params.motion_mu),1);
            for i = 1:length(pair_params.motion_mu)
                pair_mot_x_feat(i) = mean( mean( exp(-(mat_mot_x - pair_params.motion_mu(i)).^2/pair_params.motion_var(i)))) ;
            end
            
            if sum(pair_mot_x_feat)>EPS
            pair_mot_x_feat = pair_mot_x_feat/sum(pair_mot_x_feat);
            end
            
            pair_mot_y_feat = zeros(length(pair_params.motion_mu),1);
            for i = 1:length(pair_params.motion_mu)
                pair_mot_y_feat(i) =  mean( mean( exp(-(mat_mot_y - pair_params.motion_mu(i)).^2/pair_params.motion_var(i)))) ;
            end
             if sum(pair_mot_y_feat)>EPS
            pair_mot_y_feat = pair_mot_y_feat/sum(pair_mot_y_feat);
             end
            pair_mot_xy_feat = pair_mot_x_feat*pair_mot_y_feat';
            
            
            
            %             cur_pair_feat = pair_loc_t_feat(:)*pair_loc_xy_feat(:)';
            %             cur_pair_feat = cur_pair_feat(:)*pair_mot_xy_feat(:)';
            
            cur_pair_feat = [pair_loc_t_feat(:)',pair_loc_xy_feat(:)',pair_mot_xy_feat(:)'];
%             cur_pair_feat = cur_pair_feat(:)*pair_mot_xy_feat(:)';
%             size(cur_pair_feat)
           
            feat_norm = norm(cur_pair_feat);
            if feat_norm > EPS
                parts_pair_features(ind,:) = cur_pair_feat/norm(cur_pair_feat);
            else
                fprintf('zero feat norm!\n');
            end
            
            %
            
            % parts_pair_features(ind,:) = get_pair_features_mex(trajectory_i,parts_features_j,pair_params);
        end
    end


    save_name = fullfile(extract_part_features_config.path,sprintf('s%02d_c%03d_v%03d_pair.mat',cur_splits,cur_class,cur_vid) );
    save(save_name,'parts_pair_features');
    
    % -------------------------------------------------------------------------
    
else
    
    load_name = fullfile(codes_path,sprintf('codes_s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature));
    load(load_name,'codes');
    
    load_name = fullfile(feature_path,sprintf('c%03d_v%03d_info.mat',cur_class,cur_vid));
    load(load_name,'vid_info');
    
    load_name = fullfile(feature_path,sprintf('c%03d_v%03d_location.mat',cur_class,cur_vid));
    load(load_name,'location');
    
    parts_features = zeros(cluster_num,size(codes,2));
    parts_location = zeros(cluster_num,3);
    
    for i_cluster = 1:cluster_num
        
        
        switch class(cluster_id)
            case 'double'
                select_id = find(cluster_id == i_cluster);
            case 'cell'
                select_id = cluster_id{i_cluster};
            otherwise
                fprintf('WRONG!!! type of cluster_id wrong\n');
        end
        
        
        cur_location = location(select_id,:);
        
        x_low = min(cur_location(:,2));
        x_high = max(cur_location(:,2));
        
        y_low = min(cur_location(:,3));
        y_high = max(cur_location(:,3));
        
        t_low = min(cur_location(:,1));
        t_high = max(cur_location(:,1));
        
        VOI_select_id = (location(:,2) >= x_low)&(location(:,2) <= x_high)&...
            (location(:,3) >= y_low)&(location(:,3) <= y_high )&...
            (location(:,1) >= t_low)&(location(:,1) <= t_high);
        
        cur_codes = codes(VOI_select_id,:);
        
        
        %     max_loc = max(cur_location(:,1:3),[],1);
        %     min_loc = min(cur_location(:,1:3),[],1);
        
        % parts_location(i_cluster,:) = mean(cur_location(:,1:3),1);
        
        parts_location(i_cluster,:) = [(t_low+t_high)/2,(x_low+x_high)/2,(y_low+y_high)/2];
        
        switch extract_part_features_config.params.pooling
            case 'max'
                parts_features(i_cluster,:) = max(cur_codes,[],1);
                
            otherwise
                fprintf('wrong type of extract_part_features_config.params.pooling %s!\n',extract_part_features_config.params.pooling);
                returnl
                
        end
        
    end
    
    
    % normalization using l2 ----------------------
    EPS = 1e-6;
    norm_data = sqrt( sum(parts_features.^2,2 ) );
    parts_features(norm_data >EPS,:) = parts_features(norm_data >EPS,:) ./ repmat(norm_data(norm_data>EPS),1,size(parts_features,2));
    
    save_name = fullfile(extract_part_features_config.path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature) );
    save(save_name,'parts_features');
    
    save_name = fullfile(extract_part_features_config.path,sprintf('s%02d_c%03d_v%03d_location.mat',cur_splits,cur_class,cur_vid) );
    save(save_name,'parts_location');
    
end


end