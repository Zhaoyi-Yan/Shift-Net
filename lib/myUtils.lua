require 'nn'

function printNet(net)

    for i = 1, net:size(1) do
        print(string.format("%d: %s", i, net.modules[i]))
    end
    
end

function findModule(net, layer)
    assert(type(layer) == 'string', 'layer name must be string')
    for i = 1, net:size(1) do
        if tostring(net.modules[i]) == layer then
            return i
        end
    end
    return -1
end

-- Only useful for those using conv outputs as mask(need to binary)
-- If calls cal_feat_mask, then do not call binary_mask
-- It takes in conv_features to make binarized with a specific threshold.
function binary_mask(inMask, threshold)
    assert(inMask:nDimension() == 2,'mask must be 2 dimensions')
    if torch.type(inMask) ~= 'torch.CudaTensor' then
        inMask = inMask:cuda()
    end

    output = torch.gt(inMask, threshold):mul(1)
    return output
end


-- Be sure that it do not need to output byte mask,
-- just make it full of 1s and 0s.
-- Output: cuda Tensor of 2 dimension(passed into Norparameter.)
function cal_feat_mask(inMask, conv_layers, threshold)
    assert(inMask:nDimension() == 2,'mask must be 2 dimensions')
    if torch.type(inMask) ~= 'torch.CudaTensor' then
        inMask = inMask:cuda()
    end
    local lnet = nn.Sequential()
    for id_net = 1, conv_layers do
        local conv = nn.SpatialConvolution(1, 1, 4, 4, 2, 2, 1, 1):noBias()
        conv.weight:fill(1/16)
        lnet = lnet:add(conv)
    end
    lnet:cuda()
    local output = lnet:forward(inMask:view(1,inMask:size(1),inMask:size(2)))
    assert(inMask:size(1)/(torch.pow(2,conv_layers)) == output:size(2))
    output = output:squeeze()  -- remove batch_size channel

    output = torch.gt(output, threshold):cuda():mul(1)


    return output
end

-- Changes the single index to double index in Lua
function single_index2_double(index,width)
      local i = math.floor((index-1)/width)+1
      local j = index - (i-1)*width
      return i,j
end


-- mask_global should be 1*256*256 or 1*1*256*256
function create_gMask(pattern, mask_global, MAX_SIZE, opt, maxPartition)
   local maxPartition = maxPartition or 30
   if pattern == nil then
        error('..')
   end
   local temp_mask
   local mask, wastedIter
   wastedIter = 0
   while true do
     local x = torch.uniform(1, MAX_SIZE-opt.fineSize)
     local y = torch.uniform(1, MAX_SIZE-opt.fineSize)
     mask = pattern[{{y,y+opt.fineSize-1},{x,x+opt.fineSize-1}}]  -- view, no allocation
     local area = mask:sum()*100./(opt.fineSize*opt.fineSize)
     if area>20 and area<maxPartition then  -- want it to be approx 75% 0s and 25% 1s
        -- print('wasted tries: ',wastedIter)
        break
     end
     wastedIter = wastedIter + 1
   end
   if mask_global:nDimension() == 3 then
     torch.repeatTensor(mask_global, mask, 1,1,1) 
   else
     torch.repeatTensor(mask_global, mask, opt.batchSize,1,1,1)
   end

   return mask_global
end


function save_mask(mask_name, mask)
    local mask = mask:float()
    if mask:nDimension() == 4 then
        image.save(mask_name, mask[1])
    elseif mask:nDimension() == 3 then
        image.save(mask_name, mask)
    elseif mask:nDimension() == 2 then
        image.save(mask_name, torch.repeatTesor(mask,1,1,1))
    else
        error('mask dimension error')
    end
end

function load_mask(mask_name, fineSize)
    local mask = image.load(mask_name,1,'byte')
    local mask_scale = image.scale(mask, fineSize, 'simple')
    mask_scale:div(255)
    return torch.repeatTensor(mask_scale,1,1,1,1):byte()
end

-- for patch 1*1, mask_thred can only be 1
-- for patch 3*3, it can be 1~9
function cal_mask_given_mask_thred(img, mask, patch_size, stride, mask_thred)
    assert(img:nDimension() == 3, 'img has to be 3 dimenison!')
    assert(mask:nDimension() == 2, 'mask has to be 2 dimenison!')
    local nDim = img:nDimension()
    local _, H, W = img:size(nDim-2), img:size(nDim-1), img:size(nDim)
    local nH = math.floor( (H - patch_size)/stride + 1)
    local nW = math.floor( (W - patch_size)/stride + 1)
    local N = nH*nW
-- 1. flag
--    It is a flag that indicating whether the patch center in the mask
--    1: yes;  0:no.

    local flag = torch.zeros(N):long() -- need remove later
    local offsets_tmp_vec = torch.zeros(N):long() -- It is directly flatten version.
    local nonmask_point_idx_all = torch.zeros(N):long()

    local tmp_non_mask_idx = 0
    for i=1,N do
        local h = math.floor((i-1)/nW)  -- zero-index
        local w = math.floor((i-1)%nW)  -- zero-index
        -- When swap_sz is 1, the mask_tmp is only a point.
        local mask_tmp = mask[{
        {1 + h*stride, 1 + h*stride + patch_size-1},
        {1 + w*stride, 1 + w*stride + patch_size-1}
        }]

        -- If the patch is totally outside the mask region.
        -- We can use different value to adapt something.
        -- patch_size = 1 ----> mask_thred = 1
        -- patch_size = 3 ----> mask_thred = 5
        if torch.sum(mask_tmp) < mask_thred then  --outside
            tmp_non_mask_idx = tmp_non_mask_idx + 1
            nonmask_point_idx_all[tmp_non_mask_idx] = i
        else  -- in the mask
            flag[i] = 1
            offsets_tmp_vec[i] = -1
        end
    end

    local nonmask_point_idx = nonmask_point_idx_all:narrow(1, 1, tmp_non_mask_idx)

    -- get flatten_offsets
    local offset_value = nil
    local flatten_offsets = torch.LongTensor(tmp_non_mask_idx):zero()
    local flatten_offsets_all = torch.LongTensor(N):zero()
    for i = 1, N do
        offset_value = torch.sum(offsets_tmp_vec[{{1, i}}]) --It is correct(have checked!)

        -- If the point is in the mask, offset_value -1 is the real offset_value.
        -- In practise, as we set offset_value is negative, so +1.
        if flag[i] == 1 then 
            offset_value = offset_value + 1
        end
        flatten_offsets_all[i+offset_value] = -offset_value -- Should neg offset_value  (Have checked!!)
    end
    -- Then we need crop flatten_offsets_all to get flatten_offsets.
    -- If the last several elements are all masked, then these several elements
    -- are dummy, and should be cropped!
    flatten_offsets = flatten_offsets_all:narrow(1, 1, tmp_non_mask_idx)


    return flag, nonmask_point_idx, flatten_offsets
end
