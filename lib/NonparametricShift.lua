local NonparametricPatchAutoencoderShift = torch.class('NonparametricPatchAutoencoderShift')

function NonparametricPatchAutoencoderShift.buildAutoencoder(target_img, mask, patch_size, stride, normalize, interpolate, nonmask_point_idx)
    local nDim = 3
    assert(target_img:nDimension() == nDim, 'target image must be of dimension 3.')
    local patch_size = patch_size or 3
    local stride = stride or 1
    local nonmask_point_idx = nonmask_point_idx

    local type = target_img:type()
    local C = target_img:size(nDim-2)
    local patches_all, patches_part = NonparametricPatchAutoencoderShift._extract_patches(target_img, mask, patch_size, stride, nonmask_point_idx)
    local npatches_part = patches_part:size(1)
    local npatches_all = patches_all:size(1)
    local conv_enc_all, conv_enc_nonMask, conv_dec_all, conv_dec_nonMask = nil, nil, nil, nil

    local conv_enc_nonMask, conv_dec_nonMask = NonparametricPatchAutoencoderShift._build(patch_size, stride, C, patches_part, npatches_part, normalize, interpolate)--
    local conv_enc_all, conv_dec_all = NonparametricPatchAutoencoderShift._build(patch_size, stride, C, patches_all, npatches_all, normalize, interpolate)--

    return conv_enc_all, conv_enc_nonMask, conv_dec_all, conv_dec_nonMask
end

function NonparametricPatchAutoencoderShift._build(patch_size, stride , C, target_patches, npatches, normalize, interpolate)
    -- for each patch, divide by its L2 norm.
    local enc_patches = target_patches:clone()
    for i=1,npatches do
        enc_patches[i]:mul(1/(torch.norm(enc_patches[i],2)+1e-8))--< ,S>/|S|
    end

    ---- Convolution for computing the semi-normalized cross correlation ----
    local conv_enc = nn.SpatialConvolution(C, npatches, patch_size, patch_size, stride, stride):noBias()
    conv_enc.weight = enc_patches
    conv_enc.gradWeight = nil
    conv_enc.accGradParameters = __nop__
    conv_enc.parameters = __nop__

    if normalize then
        -- normalize each cross-correlation term by L2-norm of the input
        local aux = conv_enc:clone()
        aux.weight:fill(1)
        aux.gradWeight = nil
        aux.accGradParameters = __nop__
        aux.parameters = __nop__
        local compute_L2 = nn.Sequential()
        compute_L2:add(nn.Square())
        compute_L2:add(aux)
        compute_L2:add(nn.Sqrt())

        local normalized_conv_enc = nn.Sequential()
        local concat = nn.ConcatTable()
        concat:add(conv_enc)
        concat:add(compute_L2)
        normalized_conv_enc:add(concat)
        normalized_conv_enc:add(nn.CDivTable())
        normalized_conv_enc.nInputPlane = conv_enc.nInputPlane
        normalized_conv_enc.nOutputPlane = conv_enc.nOutputPlane
        conv_enc = normalized_conv_enc
    end

    ---- Backward convolution for one patch ----
    local conv_dec = nn.SpatialFullConvolution(npatches, C, patch_size, patch_size, stride, stride):noBias()
    conv_dec.weight = target_patches
    conv_dec.gradWeight = nil
    conv_dec.accGradParameters = __nop__
    conv_dec.parameters = __nop__

    -- normalize input so the result of each pixel location is a
    -- weighted combination of the backward conv filters, where
    -- the weights sum to one and are proportional to the input.
    -- the result is an interpolation of all filters.
    if interpolate then
        local aux = nn.SpatialFullConvolution(1, 1, patch_size, patch_size, stride, stride):noBias()
        aux.weight:fill(1)
        aux.gradWeight = nil
        aux.accGradParameters = __nop__
        aux.parameters = __nop__

        local counting = nn.Sequential()
        counting:add(nn.Sum(1,3))           -- sum up the channels
        counting:add(nn.Unsqueeze(1,2))     -- add back the channel dim
        counting:add(aux)
        counting:add(nn.Squeeze(1,3))
        counting:add(nn.Replicate(C,1,2))   -- replicates the channel dim C times.

        interpolating_conv_dec = nn.Sequential()
        local concat = nn.ConcatTable()
        concat:add(conv_dec)
        concat:add(counting)
        interpolating_conv_dec:add(concat)
        interpolating_conv_dec:add(nn.CDivTable())
        interpolating_conv_dec.nInputPlane = conv_dec.nInputPlane
        interpolating_conv_dec.nOutputPlane = conv_dec.nOutputPlane
        conv_dec = interpolating_conv_dec
    end

    return conv_enc, conv_dec
end


-- Here we use a custom extract_patches, mainly recording the patch 'left-top' and 'right-bottom'
-- need to check. it should work well for odd numbers.
function NonparametricPatchAutoencoderShift._extract_patches(img, mask, patch_size, stride, nonmask_point_idx)
    local nDim = 3
    assert(img:nDimension() == nDim, 'image must be of dimension 3.')

    local kH, kW = patch_size,patch_size
    local dH, dW = stride,stride
    local input_windows = img:unfold(2, kH, dH):unfold(3, kW, dW)

    local i_1, i_2, i_3, i_4, i_5 = input_windows:size(1), input_windows:size(2), input_windows:size(3), input_windows:size(4), input_windows:size(5)
    input_windows = input_windows:permute(2,3,1,4,5):contiguous():view(i_2*i_3, i_1, i_4, i_5)

    patches_all = input_windows
    patches = input_windows:index(1, nonmask_point_idx) --It returns a new tensor, it is nonmask patches!

    return patches_all, patches
end


function __nop__()
    -- do nothing
end
