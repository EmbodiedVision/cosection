%
% This file is part of Co-Section.
%
% Copyright (C) 2020 Max-Planck-Gesellschaft.
% Developed by Michael Strecke <mstrecke at tue dot mpg dot de>.
% For more information see <https://cosection.is.tue.mpg.de/>.
% If you use this code, please cite the respective publication as
% listed on the website.
%
% Co-Section is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% Co-Section is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Co-Section.  If not, see <https://www.gnu.org/licenses/>.
%

%% Evaluate meshes

result_path = '/work/test_cosection/';
reference_path = 'data/';
for obj_name = {'truck', 'car'} %, 'plane'}
obj_name = obj_name{:}
fprintf("Processing %s\n", obj_name)

reference_path = [reference_path, obj_name '_ref.ply'];
[reference.vertices, reference.faces] = read_ply(reference_path);


reference = remove_nonref_verts ( reference );

switch obj_name
    case 'truck'
        obj_id = 2;
    case 'car'
        obj_id = 4;
    case 'plane'
        obj_id = 1;
end

for idx=1:5
    switch idx
    case 1
        appr='tsdf';
    case 2
        appr='baseline';
    case 3
        appr='hull';
    case 4
        appr='inter';
    case 5
        appr='full';
    end
    
    if strcmp(appr, 'tsdf')
        mesh_path = [result_path, 'car4_full/mesh_', int2str(obj_id), '.ply'];
    else
        mesh_path = [result_path, 'car4_full/mesh_', int2str(obj_id), '.ply'];
    end

    [mesh(idx).vertices,mesh(idx).faces] = read_ply(mesh_path);
    mesh(idx) = remove_nonref_verts ( mesh(idx) );


    tic
    acc{idx} = point2trimesh ( reference, 'QueryPoints', mesh(idx).vertices, 'Algorithm', 'parallel' );
    toc
    tic
    compl{idx} = point2trimesh ( mesh(idx), 'QueryPoints', reference.vertices, 'Algorithm', 'parallel' );
    toc
end
%%
maxval = 0;

minp = [inf inf inf];
maxp = [-inf -inf -inf];
for idx = 1:5
    maxacc = max(acc{idx});
    if maxacc > maxval
        maxval = maxacc;
    end
    maxcompl = max(compl{idx});
    if maxcompl > maxval
        maxval = maxcompl;
    end

    minp = min(min(mesh(idx).vertices,[],1), minp);
    maxp = max(max(mesh(idx).vertices,[],1), maxp);
end
%%
for idx=1:5
    switch idx
    case 1
        appr='tsdf';
    case 2
        appr='baseline';
    case 3
        appr='hull';
    case 4
        appr='inter';
    case 5
        appr='full';
    end
%     dlmwrite([result_path, 'acc_', obj_name, '_', outname, '.txt'], acc{idx})
%     dlmwrite([result_path, 'comp_', obj_name, '_', outname, '.txt'], compl{idx})
    figure(1),
    switch obj_name
        case 'car'
            pcshow ( mesh(idx).vertices .* [1, -1, -1], abs(acc{idx}) ), % for car
        case 'truck'
            pcshow ( ([1 0 0; 0 0 1; 0 -1 0] * (mesh(idx).vertices .* [-1, 1, -1])')', abs(acc{idx}) ), % for truck
    end
    title ("Accuracy"),
    set(gcf,'color','w'),
    set(gca,'color','w'),
    set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15]),
    set(gca, 'FontSize', 8.5);
    colorbar('FontSize', 9.5)
    caxis ([0,maxval])
    xlim([minp(1), maxp(1)])
    ylim([minp(2), maxp(2)])
    zlim([minp(3), maxp(3)])
    set(gcf,'Units','inches');
    screenposition = get(gcf,'Position');
    set(gcf,...
        'PaperPosition',[0 0 screenposition(3:4)],...
        'PaperSize', [screenposition(3:4)])
    print( '-dpng', '-r600', [result_path, '/num_eval/acc_', obj_name, '_', appr])
    figure(2),
    switch obj_name
        case 'car'
            pcshow ( reference.vertices .* [1, -1, -1], abs(compl{idx}) ), % for car
        case 'truck'
            pcshow ( ([1 0 0; 0 0 1; 0 -1 0] * (reference.vertices .* [-1, 1, -1])')', abs(compl{idx}) ); % for truck
    end
    title ("Completeness"),
    set(gcf,'color','w'),
    set(gca,'color','w'),
    set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
    set(gca, 'FontSize', 8.5)
    colorbar('FontSize', 9.5)
    caxis ([0,maxval])
    xlim([minp(1), maxp(1)])
    ylim([minp(2), maxp(2)])
    zlim([minp(3), maxp(3)])
    set(gcf,'Units','inches');
    screenposition = get(gcf,'Position');
    set(gcf,...
        'PaperPosition',[0 0 screenposition(3:4)],...
        'PaperSize', [screenposition(3:4)]);
    print( '-dpng', '-r600', [result_path, '/num_eval/comp_', obj_name, '_', appr])
end
save([result_path, obj_name])
end




function [clean_mesh] = remove_nonref_verts ( mesh )
validVerts = false ( size ( mesh.vertices, 1 ), 1 );
reduceIdx = zeros ( size ( mesh.faces ) );

for idx=1:size(mesh.vertices,1)
    validVerts(idx) = any(any( mesh.faces == idx ));
    if (~validVerts(idx))
        reduceIdx( mesh.faces > idx ) = reduceIdx( mesh.faces > idx ) + 1;
    end
end

clean_mesh.vertices = mesh.vertices ( validVerts,: );
clean_mesh.faces = mesh.faces - reduceIdx;

if size(clean_mesh.vertices,1) ~= size(mesh.vertices,1)
    fprintf ( "Removed %d invalid vertices!\n", size(mesh.vertices,1) - size(clean_mesh.vertices,1) );
else
    fprintf ( "Mesh already clean!\n" );
end


end
