% show clusters
function return_state = m_show_clusters_v0(global_config)
return_state = 1;


% set the params

show_class_idx = global_config.show.show_class_idx;
showed_tr_num = global_config.show.showed_tr_num;
feature_types = {'hof','hog','shape','mbhx','mbhy'};


% load dateset info
load(global_config.read_dataset_info.file_name);
% 'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num', 'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class'

splits = [1];
do_vids = cell(length(splits),class_num);
switch class(show_class_idx)
    case 'cell'
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(show_class_idx)
                
                do_vids{cur_splits,show_class_idx{i}(1)} = [do_vids{cur_splits,show_class_idx{i}(1)};show_class_idx{i}(2)];
            end
            
        end
    case 'double'
        
        for i_splits = 1:length(splits)
            cur_splits = splits(i_splits);
            for i = 1:length(show_class_idx)
                do_vids{cur_splits,show_class_idx(i)} = [train_set_idx{cur_splits}{show_class_idx(i)};test_set_idx{cur_splits}{show_class_idx(i)}];
            end
        end
        
    otherwise
        fprintf('wrong type of global_config.extract_features.class_idx!\n');
        return_state = 0;
        return;
end

response_colors = [
1,0,0;
0,1,0;
0.2,1,0;
0,0,1;
0.1,0,1;
0,0.1,1
];


fig = figure;

for i_splits = 1:length(splits)
    cur_splits = splits(i_splits);
    
    feat_num = length(feature_types);
    
    
    
    for i_class = 1:class_num
        
        cur_class = i_class;
        if length(do_vids{cur_splits,cur_class}) == 0
            continue;
        end
        
        W_all = cell(feat_num,3);
        % load all models
        for i_feat = 1:feat_num
            
            cur_feature = feature_types{i_feat};
            
            
            cur_feature_1 = [cur_feature,'-1'];
            load_name = fullfile(global_config.learn_action_parts.path,sprintf('s%02d_c%03d_%s.mat',cur_splits,cur_class,cur_feature_1));
            load( load_name, 'action_parts_model');
            
            W_1 = action_parts_model;
            
            
            
            cur_feature_2 = [cur_feature,'-2'];
            load_name = fullfile(global_config.learn_action_parts.path,sprintf('s%02d_c%03d_%s.mat',cur_splits,cur_class,cur_feature_2));
            load( load_name, 'action_parts_model');
            
            W_2 = action_parts_model;
            
            
            
            cur_feature_3 = [cur_feature,'-3'];
            
            load_name = fullfile(global_config.learn_action_parts.path,sprintf('s%02d_c%03d_%s.mat',cur_splits,cur_class,cur_feature_3));
            load( load_name, 'action_parts_model');
            
            W_3 = action_parts_model;
            
            W_all{i_feat,1} = W_1;
            W_all{i_feat,2} = W_2;
            W_all{i_feat,3} = W_3;
            
        end
        
        
        cur_vid_idx_total = do_vids{cur_splits,cur_class};
        
        %         if length(cur_vid_idx_total) > 5
        %             rand_select_idx = randperm(length(cur_vid_idx_total),5);
        %             cur_vid_idx_select = cur_vid_idx_total(rand_select_idx);
        %         else
        %             cur_vid_idx_select = cur_vid_idx_total;
        %         end
        
        cur_vid_idx_select = cur_vid_idx_total;
        
        for i_vid = 1:length(cur_vid_idx_select)
            cur_vid = cur_vid_idx_select(i_vid);
            
            if sum(train_set_idx{cur_splits}{cur_class} == cur_vid)
                train_or_test = 'train';
            elseif sum(test_set_idx{cur_splits}{cur_class} == cur_vid)
                train_or_test = 'test';
            else
                train_or_test = 'none';
            end
            
            
            fprintf('class:%3d video:%3d\n',cur_class,cur_vid);
            video_path = fullfile(global_config.dataset_path, vid_paths{cur_class}{cur_vid});
            % call exe to show the trajectory video
            extract_fea_fun = '..\common\exe_functions\get_dt_feature_nm_show.exe';
            
            temp_feature_name = sprintf('temp_feature_c%03d_v%03d',cur_class,cur_vid);
            temp_vid_info_name = sprintf('temp_vid_info_c%03d_v%03d',cur_class,cur_vid);
            
            % call exe to get temp file
            %                 cmd_str = [extract_fea_fun ' '  video_path ' ' temp_feature_name ' ' temp_vid_info_name];
            %                 dos(cmd_str);
            
            % load video ----------------------------------------------
            
            raw_video = VideoReader(video_path);
            
            frame_num = raw_video.NumberOfFrames;
            vid_height = raw_video.Height;
            vid_width = raw_video.Width;
            
            mov(1:frame_num) = struct('cdata', zeros(vid_height, vid_width, 3, 'uint8'), 'colormap', []);
            
            % find the cluster related
            load_name = fullfile(global_config.clustering.path,sprintf('clusters_c%03d_v%03d.mat',cur_class,cur_vid));
            load(load_name,'cluster_id');
            
            % show the clusters
            feature_path = global_config.extract_features.path;
            load_name = fullfile(feature_path, sprintf('c%03d_v%03d_location',cur_class,cur_vid));
            load(load_name,'location');
            
            load_name = fullfile(feature_path, sprintf('c%03d_v%03d_trajectory',cur_class,cur_vid));
            load(load_name,'trajectory');
            
            load_name = fullfile(feature_path, sprintf('c%03d_v%03d_info',cur_class,cur_vid));
            load(load_name,'vid_info');
            
            t_begin = location(:,1);
            x = trajectory(:,1:2:end);
            y = trajectory(:,2:2:end);
            
            dt_len = size(x,2);
            
            t_begin = t_begin - dt_len +2;
            
            t = t_begin;
            for i_t = 1:dt_len-1
                t = [t,t_begin+i_t];
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % infer response
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [scores,response] = infer_resonse(W_all,cur_splits,cur_class,cur_vid,feature_types,global_config);
            
            
            
            
            cluster_id_cell = cluster_id;
            clear cluster_id;
            cluster_id = zeros(size(location,1),1);
            for i = 1:length(cluster_id_cell)
                cluster_id(cluster_id_cell{i}) = i;
            end
            
            cluster_num = length(cluster_id_cell);
            % sampling
            non_clustered_idx = find(cluster_id == 0);
            
            clusters_x = cell(cluster_num+1,1);
            clusters_y = cell(cluster_num+1,1);
            clusters_t = cell(cluster_num+1,1);
            
            for i_cluster = 1:cluster_num
                cur_cluster_id = cluster_id_cell{i_cluster};
                cur_cluster_size = length(cur_cluster_id);
                if cur_cluster_size > showed_tr_num
                    cur_cluster_id = cur_cluster_id(randperm(cur_cluster_size,showed_tr_num));
                end
                
                clusters_x{i_cluster} = x(cur_cluster_id,:);
                clusters_y{i_cluster} = y(cur_cluster_id,:);
                clusters_t{i_cluster} = t(cur_cluster_id,:);
            end
            
            
            
            
            for i_feat = 1:length(feature_types)
                
                cur_feat = feature_types{i_feat};
                clusters_show = [];
                for i_node = 1:size(response,2)
                    clusters_show = [clusters_show,response{i_feat,i_node}];
                end
                
                clustered_video = fullfile(global_config.local_path,'show_cluster',sprintf('%s_c%03d_v%03d_cluster_%s.avi',train_or_test,cur_class,cur_vid,cur_feat));
                if ~exist(fullfile(global_config.local_path,'show_cluster'),'dir')
                    mkdir(fullfile(global_config.local_path,'show_cluster'));
                end
                video_recorder = VideoWriter(clustered_video);
                open(video_recorder);
                
                % given different cluster different color
                
                
                fprintf('%3d/%3d\n',1,1);
                
                for i_frame = 1:frame_num-1
                    
                    fprintf('%c%c%c%c%c%c%c%c\n',8,8,8,8,8,8,8,8,8);
                    fprintf('%3d/%3d\n',i_frame,frame_num);
                    
                    mov(i_frame).cdata = read(raw_video, i_frame);
                    imshow(mov(i_frame).cdata);
                    
                    
                    for i_cluster = 1:length(clusters_show)
                        
                        cur_color = response_colors(i_cluster,:);
                        cur_frame_ind = clusters_t{i_cluster} == i_frame;
                        
                        if ~any(cur_frame_ind(:))
                            continue;
                        end
                        
                        cur_x = clusters_x{i_cluster}(cur_frame_ind);
                        cur_y = clusters_y{i_cluster}(cur_frame_ind);
                
                        
                        rectangle('Position',[cur_x,cur_y,1,1],'EdgeColor',cur_color,'FaceColor',cur_color);
                                                
                        % line(x(dt_idx_cur_frame(i_dt),1:dt_end_frame(i_dt))',y(dt_idx_cur_frame(i_dt),1:dt_end_frame(i_dt))','Color',cur_color);
                        
                        % plot the trajectory
                        %                         if dt_end_frame(i_dt) > 1
                        %                             for i_end = 1:dt_end_frame(i_dt)-1
                        %                                 line(x(dt_idx_cur_frame(i_dt),i_end:i_end+1)',y(dt_idx_cur_frame(i_dt),i_end:i_end+1)','Color',cur_color*(i_end)/(dt_end_frame(i_dt)-1));
                        %                             end
                        %                         end
                        
                        
                    end
                                      
                    MOV = getframe(fig);
                    writeVideo(video_recorder,MOV);
                    
                    
                    %                      pause(0.03);
                    
                end
                
                close(video_recorder);
                clear video_recorder;
                
                
                clear raw_video;
                
            end
        end % i_vid ends
        
    end  % iclass ends
    
    
end % i_split



end



%%
function [scores,response] = infer_resonse(W_all,cur_splits,cur_class,cur_vid,feature_types,global_config)

response = cell(size(W_all));
scores = zeros(size(W_all));

part_features_path = global_config.extract_part_features.path;

load_name = fullfile(part_features_path,sprintf('s%02d_c%03d_v%03d_pair.mat',cur_splits,cur_class,cur_vid) );
load(load_name,'parts_pair_features');

global_config.learn_action_parts.struct_M5IL.graph.edge_num = global_config.learn_action_parts.struct_M5IL.graph.node_num*(global_config.learn_action_parts.struct_M5IL.graph.node_num-1)/2;
load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_pair.mat',1,1,1) );
load(load_name,'parts_pair_features');
global_config.learn_action_parts.struct_M5IL.graph.edge_dim = size(parts_pair_features,2);

load_name = fullfile(global_config.extract_part_features.path, sprintf('s%02d_c%03d_v%03d_%s.mat',1,1,1,'shape') );
load(load_name,'parts_features');
global_config.learn_action_parts.struct_M5IL.graph.node_dim = size(parts_features,2)+1;

global_config.learn_action_parts.struct_M5IL.graph.psi_dim = global_config.learn_action_parts.struct_M5IL.graph.node_dim * global_config.learn_action_parts.struct_M5IL.graph.node_num + global_config.learn_action_parts.struct_M5IL.graph.edge_dim * global_config.learn_action_parts.struct_M5IL.graph.edge_num;


for i_feat = 1:length(feature_types)
    cur_feature = feature_types{i_feat};
    load_name = fullfile(part_features_path,sprintf('s%02d_c%03d_v%03d_%s.mat',cur_splits,cur_class,cur_vid,cur_feature) );
    load(load_name,'parts_features');
    
    for i_node = 1:size(W_all,2)
        
        if i_node == 1
            
            responds_data = parts_features * W_all{i_feat,i_node};
            responss_data_max = max(responds_data,[],2);
            [scores(i_feat,i_node),response{i_feat,i_node}]= max(responss_data_max);
            
        else
            
            channel_num = struct_M5IL_config.channel_num;
            
            phi_nodes_all = cell(channel_num,1);
            phi_edges_all = cell(channel_num,1);
            
            channel_num = 3;
            for i_channel = 1:channel_num
                
                
                [W_nodes,W_edges] = get_graph_W (W_all{i_feat,i_node}(:,i_channel),global_config.learn_action_parts.struct_M5IL);
                
                phi_nodes_all{i_channel} = parts_features*W_nodes;
                phi_edges_all{i_channel} = parts_pair_features*W_edges;
                
            end
            
            
            if i_node == 2
                global_config.learn_action_parts.struct_M5IL.method = 'traversal';
            elseif i_node == 3
                global_config.learn_action_parts.struct_M5IL.method = 'TRW-S';
            end
            
            [scores(i_feat,i_node),response{i_feat,i_node}] = infer_bag_score(phi_nodes_all,phi_edges_all,global_config.learn_action_parts.struct_M5IL);
            
        end
        
        
    end
    
end

end

%%
function [W_nodes,W_edges] = get_graph_W (W,config)

W_nodes = reshape(W(1:config.graph.node_dim*config.graph.node_num),config.graph.node_dim,config.graph.node_num);

W_edges = reshape(W(1+config.graph.node_dim*config.graph.node_num:end),config.graph.edge_dim,config.graph.edge_num);

end

%%
function [scores,node_ass_idx] = infer_bag_score(phi_nodes_all,phi_edges_all,config)

switch config.method
    case 'TRW-S'
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i_channel = 1:channel_num
            
            
            
            phi_nodes_all{i_channel} = -phi_nodes_all{i_channel};
            phi_edges_all{i_channel} = -phi_edges_all{i_channel};
            
        end
        
        [scores,node_ass_idx] = MRF_infer_TRW_S_mex(phi_nodes_all,phi_edges_all,config.channel_num);
        
    case 'traversal'
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         channel_num = config.channel_num;
        %         save('debug_MRF_infer_traversal.mat','phi_nodes_all','phi_edges_all','channel_num');
        [scores,node_ass_idx] = MRF_infer_traversal_mex(phi_nodes_all,phi_edges_all,config.channel_num);
    case 'node-only'
        
        scores = zeros(1,config.graph.node_num);
        node_ass_idx_all = zeros(config.channel_num,config.graph.node_num);
        for i_channel = 1:config.channel_num
            [max_val,node_ass_idx_all(i_channel,:)] = max( phi_nodes_all {i_channel} ,[],1);
            scores(i_channel) = sum(max_val);
        end
        [~,channel_max_score] = max(scores);
        node_ass_idx = node_ass_idx_all(channel_max_score,:);
        
    otherwise
        fprintf('WRONG!!! the infer method not exist!\n');
        return;
end



end
