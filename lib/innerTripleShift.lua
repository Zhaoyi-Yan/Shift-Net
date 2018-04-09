require 'lib/NonparametricShift'

local InnerTripleShift, parent = torch.class('nn.InnerTripleShift', 'nn.Module')

function InnerTripleShift:__init(cls, shift_size, threshold, stride, mask_thred, triple_weight, fixed_mask)
    parent.__init(self)

    self.cls = cls
    self.shift_size = shift_size
    self.threshold = threshold
    self.stride = stride
    self.mask_thred = mask_thred
    self.triple_w = triple_weight or 1
    self.interplote = false
    if self.shift_size ~= 1 then
        error('Haven\'t implemented patchsize>1 for now.')
        self.interplote = true
    end
    -- whether the mask is fixed_mask.(When using random mask(not fixed), you should set it false!)
    self.fixed_mask = fixed_mask
end


-- Use latter non-mask region as style to replace former masked region.
-- The last 1/3 part is shifted feature.
function InnerTripleShift:updateOutput(input)
    assert(input:nDimension() == 4)
    self.bz = input:size(1)
    self.h = input:size(3)
    self.w = input:size(4)
    self.c_real = input:size(2)

    self.c = self.c_real - 1

    local former_all = input:narrow(2, 1, self.c/2)
    local latter_all = input:narrow(2, self.c/2 + 1, self.c/2)
    local swapped_latter_all = torch.Tensor(self.bz, self.c/2, self.h, self.w):typeAs(input)

    -- Get the mask and inv_mask, they are all one channel.
    local mask_float = input:narrow(2, self.c_real, 1):narrow(1,1,1):squeeze()  -- 2D
    local mask = binary_mask(mask_float, self.threshold)

    self.ex_mask = mask:byte():repeatTensor(1, self.c/2 ,1,1)
    self.inv_ex_mask = torch.add(self.ex_mask:float():neg(), 1):byte()

    -- useful for guidance loss constraint.
    self.latent_in_mask = latter_all:clone()
    self.latent_in_mask[self.inv_ex_mask:repeatTensor(self.bz, 1, 1, 1)] = 0 -- Get the encoded latent from skip connections.

    -- Only `fixed_mask = true` and we have calculated the flag and mask_point_idx before, then we do not need to calculate them again!
    -- [TO DO: make it computation efficient.]
    if self.fixed_mask and self.cal_fixedFlag == false then
        -- has done
        assert(self.flag ~= nil, 'Lack of \'self.flag!\'')
    else
        if self.cal_fixedFlag then
            print('calculate self.flag')
        end

        self.flag, self.nonmask_point_idx, self.flatten_offsets = cal_mask_given_mask_thred(latter_all:narrow(1,1,1):squeeze(), mask, self.shift_size, self.stride, self.mask_thred)

        self.cal_fixedFlag = false
    end

    -- We need to store 'ind_lst' to be used in BP.
    self.ind_lst = torch.LongTensor(self.bz, self.h*self.w)

    for idx = 1, self.bz do
        local latter = latter_all:narrow(1, idx, 1)
        local former = former_all:narrow(1, idx, 1)

        local conv_enc, conv_new_dec
        _, conv_enc, conv_new_dec, _ = NonparametricPatchAutoencoderShift.buildAutoencoder(
            latter:squeeze(), mask, self.shift_size, self.stride, false, self.interplote, self.nonmask_point_idx)

                                                                    
        local maxcoor = nn.MaxCoord():cuda()
        conv_enc:cuda()

        -- here latter mask region should be replaced with latter non-masked region. 
        local cosine_distance = conv_enc:forward(former)

        local kbar, ind
        kbar, ind = maxcoor:forward(cosine_distance)
        conv_new_dec:cuda()

        -- calulate the real kbar and real ind.
        local real_npatches = kbar:size(2) + torch.sum(self.flag)

        local kbar_c = kbar:size(2)
        local kbar_h = kbar:size(3)
        local kbar_w = kbar:size(4)

        kbar = torch.Tensor(1, real_npatches, kbar_h, kbar_w):typeAs(kbar):zero()


        for i = 1, kbar_h do
            for j = 1, kbar_w do
                local index = (i-1)*kbar_w + j
                local non_r_ch = ind[index]
                -- now need we need to find the index of `non_r_ch`-th point in the whole map.
                local offset = self.flatten_offsets[non_r_ch]

                -- get the corrected channel, set 0 to 1
                local correct_ch = non_r_ch + offset
                kbar[{{},{correct_ch},{i},{j}}] = 1
                ind[index] = correct_ch
            end
        end

        local result_tmp = conv_new_dec:forward(kbar)  

        swapped_latter_all[idx] = result_tmp
        self.ind_lst[idx] = ind
    end

    -- Mask the swapped features.
    swapped_latter_all[self.inv_ex_mask:repeatTensor(self.bz, 1, 1, 1)] = 0


    -- construct final self.output
    self.output = torch.cat({former_all, latter_all, swapped_latter_all}, 2)
    return self.output
end



function InnerTripleShift:updateGradInput(input, gradOutput)
    self.gradInput = torch.Tensor():typeAs(input):resizeAs(input):zero()

    local c = gradOutput:size(2)
    local grad_former_all = gradOutput[{{},{1, c/3},{},{}}]
    local grad_latter_all = gradOutput[{{},{c/3+1, c*2/3},{},{}}]
    local grad_swapped_all = gradOutput[{{},{c*2/3+1, c},{},{}}]

    -- Start constructing `W`
    local spatialSize = self.h * self.w

    for idx = 1, self.bz do
        local W_mat = torch.Tensor(spatialSize, spatialSize):typeAs(input):zero()

        -- for each line of `W`
        for cnt = 1, spatialSize do
            -- It means this pixel is in the mask, and this line(index: cnt_th) 
            -- should be one-hot vector, with the `indS_th` be 1.
            if self.flag[cnt] == 1 then
                local indS = self.ind_lst[idx][cnt]
                W_mat[{cnt, indS}] = 1
            end
        end
        -- After `W` has constructed, then transpose `W`
        local W_mat_t = W_mat:t()  -- We don't need W again, so sharing the same storage is fine.

        -- view(c/3,-1):t() makes each line be a gradient of certain position which is c/3 channels.
        local grad_swapped_weighted = torch.mm(W_mat_t, grad_swapped_all[idx]:view(c/3, -1):t())

        -- Then transpose it back
        grad_swapped_weighted = grad_swapped_weighted:t():contiguous():view(1, c/3, self.h, self.w)

        grad_latter_all[idx] = torch.add(grad_latter_all[idx], grad_swapped_weighted:mul(self.triple_w))
    end

    -- zero the mask_dimension to make it no updating!
    local mask_dim = torch.Tensor(self.bz, 1, self.h, self.w):zero():typeAs(input)
    self.gradInput = torch.cat({grad_former_all, grad_latter_all, mask_dim}, 2)
    return self.gradInput
end


function InnerTripleShift:clearState()
    self.output = self.output.new()
    self.gradInput = self.gradInput.new()

end

function InnerTripleShift:setCalculateFlagTrue()
    self.cal_fixedFlag = true
end

function InnerTripleShift:setCalculateFlagFalse()
    self.cal_fixedFlag = false
end
