import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Set;

class Solution {

    // 两数之和
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>(); // 创建一个哈希表
        for (int i = 0; i < nums.length; ++i) { // 遍历数组
            int complement = target - nums[i]; // 计算目标值与当前值的差值
            if (map.containsKey(complement)) { // 如果哈希表中包含该差值
                return new int[] { map.get(complement), i }; // 返回差值的索引和当前索引
            }
            map.put(nums[i], i); // 将当前值和索引存入哈希表
        }
        throw new IllegalArgumentException("No two sum solution"); // 如果找不到解决方案，则抛出异常
    }

    // 寻找最长无重复字符的子串
    public int lengthOfLongestSubstring(String s) {
        int[] cnt = new int[128]; // 创建一个长度为128的数组，用于存储字符出现的次数
        int ans = 0, n = s.length(); // 初始化答案为0，字符串长度为n
        for (int l = 0, r = 0; r < n; ++r) { // 遍历字符串
            char c = s.charAt(r); // 获取当前字符
            ++cnt[c]; // 增加当前字符的出现次数
            while (cnt[c] > 1) { // 如果当前字符的出现次数大于1
                --cnt[s.charAt(l++)]; // 减少左侧字符的出现次数，并移动左指针
            }
            ans = Math.max(ans, r - l + 1); // 更新答案
        }
        return ans; // 返回答案
    }

    // 寻找两个有序数组的中位数
    private int m; // 数组1的长度
    private int n; // 数组2的长度
    private int[] nums1; // 数组1
    private int[] nums2; // 数组2

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        m = nums1.length; // 数组1的长度
        n = nums2.length; // 数组2的长度
        this.nums1 = nums1; // 数组1
        this.nums2 = nums2; // 数组2
        int a = f(0, 0, (m + n + 1) / 2); // 计算中位数
        int b = f(0, 0, (m + n + 2) / 2); // 计算中位数
        return (a + b) / 2.0; // 返回中位数
    }

    // 寻找两个有序数组的中位数
    private int f(int i, int j, int k) {
        if (i >= m) { // 如果数组1已经遍历完
            return nums2[j + k - 1]; // 返回数组2的第k个元素
        }
        if (j >= n) { // 如果数组2已经遍历完
            return nums1[i + k - 1]; // 返回数组1的第k个元素
        }
        if (k == 1) { // 如果k为1
            return Math.min(nums1[i], nums2[j]); // 返回两个数组中较小的元素
        }
        int p = k / 2; // 计算k的一半
        int x = i + p - 1 < m ? nums1[i + p - 1] : 1 << 30; // 计算数组1的第p个元素
        int y = j + p - 1 < n ? nums2[j + p - 1] : 1 << 30; // 计算数组2的第p个元素
        return x < y ? f(i + p, j, k - p) : f(i, j + p, k - p); // 递归调用f函数
    }

    // 合并有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        for (int i = m - 1, j = n - 1, k = m + n - 1; j >= 0; --k) { // 从后往前遍历
            nums1[k] = i >= 0 && nums1[i] > nums2[j] ? nums1[i--] : nums2[j--]; // 如果nums1的第i个元素大于nums2的第j个元素，则将nums1的第i个元素赋值给nums1的第k个元素，否则将nums2的第j个元素赋值给nums1的第k个元素
        }
    }

    // 移除元素
    public int removeElement(int[] nums, int val) {
        int k = 0; // 初始化k为0
        for (int x : nums) { // 遍历数组
            if (x != val) { // 如果当前元素不等于val
                nums[k++] = x; // 将当前元素赋值给nums的第k个元素，并增加k
            }
        }
        return k; // 返回k
    }

    // 移除有序数组中的重复元素
    public int removeDuplicates(int[] nums) {
        int k = 0; // 初始化k为0
        for (int x : nums) { // 遍历数组
            if (k == 0 || nums[k - 1] != x) { // 如果k为0，或者nums的第k-1个元素不等于x
                nums[k++] = x; // 将当前元素赋值给nums的第k个元素，并增加k
            }
        }
        return k; // 返回k
    }

    // 移除有序数组中的重复元素
    public int removeDuplicates2(int[] nums) {
        int k = 0; // 初始化k为0
        for (int x : nums) { // 遍历数组
            if (k < 2 || nums[k - 2] != x) { // 如果k小于2，或者nums的第k-2个元素不等于x
                nums[k++] = x; // 将当前元素赋值给nums的第k个元素，并增加k
            }
        }
        return k; // 返回k
    }

    // 寻找多数元素
    public int majorityElement(int[] nums) {
        int cnt = 0, m = 0; // 初始化cnt为0，m为0
        for (int x : nums) { // 遍历数组
            if (cnt == 0) { // 如果cnt为0
                m = x; // 将当前元素赋值给m
                cnt = 1; // 将cnt赋值为1
            } else { // 否则
                cnt += (x == m) ? 1 : -1; // 如果当前元素等于m，则增加cnt，否则减少cnt
            }
        }
        return m; // 返回m
    }

    // 旋转数组
    private int[] nums; // 数组

    public void rotate(int[] nums, int k) {
        this.nums = nums; // 将nums赋值给nums
        int n = nums.length; // 数组长度
        k %= n; // 取模
        reverse(0, n - 1); // 反转整个数组
        reverse(0, k - 1); // 反转前k个元素
        reverse(k, n - 1); // 反转后n-k个元素
    }

    private void reverse(int i, int j) {
        for (; i < j; ++i, --j) { // 遍历数组
            int t = nums[i]; // 交换两个元素
            nums[i] = nums[j]; // 将nums的第i个元素赋值给nums的第j个元素
            nums[j] = t; // 将nums的第j个元素赋值给nums的第i个元素
        }
    }

    // 买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        int ans = 0, mi = prices[0]; // 初始化ans为0，mi为prices的第0个元素
        for (int v : prices) { // 遍历数组
            ans = Math.max(ans, v - mi); // 更新ans
            mi = Math.min(mi, v); // 更新mi
        }
        return ans; // 返回ans
    }

    // 买卖股票的最佳时机 II
    public int maxProfit2(int[] prices) {
        int ans = 0; // 初始化ans为0
        for (int i = 1; i < prices.length; ++i) { // 遍历数组
            ans += Math.max(0, prices[i] - prices[i - 1]); // 更新ans
        }
        return ans; // 返回ans
    }

    // 买卖股票的最佳时机 III
    public int maxProfit3(int[] prices) {
        // f1 第一次买入
        // f2 第一次卖出
        // f3 第二次买入
        // f4 第二次卖出
        int f1 = -prices[0], f2 = 0, f3 = -prices[0], f4 = 0; // 初始化f1, f2, f3, f4
        for (int i = 1; i < prices.length; ++i) { // 遍历数组
            f1 = Math.max(f1, -prices[i]); // 更新f1
            f2 = Math.max(f2, f1 + prices[i]); // 更新f2
            f3 = Math.max(f3, f2 - prices[i]); // 更新f3
            f4 = Math.max(f4, f3 + prices[i]); // 更新f4
        }
        return f4; // 返回f4
    }

    // 跳跃游戏
    public boolean canJump(int[] nums) {
        int mx = 0; // 初始化mx为0
        for (int i = 0; i < nums.length; ++i) { // 遍历数组
            if (i > mx) { // 如果i大于mx
                return false; // 返回false
            }
            mx = Math.max(mx, i + nums[i]); // 更新mx
        }
        return true; // 返回true
    }

    // 跳跃游戏 II
    public int jump(int[] nums) {
        int ans = 0, mx = 0, last = 0; // 初始化ans为0，mx为0，last为0
        for (int i = 0; i < nums.length - 1; ++i) { // 遍历数组
            mx = Math.max(mx, i + nums[i]); // 更新mx
            // 当i到达last时，更新last为mx，必须跳一步
            if (i == last) {
                last = mx; // 更新last
                ++ans; // 跳一步
            }
        }
        return ans; // 返回ans
    }

    // H指数
    public int hIndex(int[] citations) {
        Arrays.sort(citations); // 将论文被引次数从小到大排序
        int n = citations.length; // 数组长度
        for (int h = n; h > 0; --h) { // 从后往前遍历
            if (citations[n - h] >= h) { // 如果当前论文被引次数大于等于h，则返回h
                return h; // 返回h
            }
        }
        return 0; // 返回0
    }

    // O(1)时间插入、删除和获取随机元素
    static class RandomizedSet {
        private Map<Integer, Integer> d = new HashMap<>(); // 存储元素及其索引
        private List<Integer> q = new ArrayList<>(); // 存储元素
        private Random rnd = new Random(); // 随机数生成器

        public RandomizedSet() {
        }

        public boolean insert(int val) {
            if (d.containsKey(val)) { // 如果元素已存在
                return false; // 返回false
            }
            d.put(val, q.size()); // 存储元素及其索引
            q.add(val); // 添加元素
            return true; // 返回true
        }

        public boolean remove(int val) {
            if (!d.containsKey(val)) { // 如果元素不存在
                return false; // 返回false
            }
            int i = d.get(val); // 获取元素的索引
            int last = q.get(q.size() - 1); // 获取最后一个元素
            q.set(i, last); // 将最后一个元素赋值给第i个元素
            d.put(last, i); // 存储最后一个元素及其索引
            q.remove(q.size() - 1); // 删除最后一个元素
            d.remove(val); // 删除元素
            return true; // 返回true
        }

        public int getRandom() {
            return q.get(rnd.nextInt(q.size())); // 返回随机元素
        }
    }

    // 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length; // 数组长度
        int[] ans = new int[n]; // 存储结果
        for (int i = 0, left = 1; i < n; ++i) { // 遍历数组
            ans[i] = left; // 存储左侧乘积
            left *= nums[i]; // 更新左侧乘积
        }
        for (int i = n - 1, right = 1; i >= 0; --i) { // 遍历数组
            ans[i] *= right; // 左侧乘积*右边的乘积
            right *= nums[i]; // 更新右边的乘积
        }
        return ans; // 返回结果
    }

    // 加油站
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int n = gas.length; // 数组长度
        int i = n - 1, j = n - 1; // 初始化i和j
        int cnt = 0, s = 0; // 初始化cnt和s
        while (cnt < n) { // 如果cnt小于n
            s += gas[j] - cost[j]; // 累计j站的净油量
            ++cnt; // 增加cnt
            j = (j + 1) % n; // j右移
            while (s < 0 && cnt < n) { // 如果净油量小于0，则i左移
                --i; // 减少i
                s += gas[i] - cost[i]; // 增加净油量
                ++cnt; // 增加cnt
            }
        }
        return s >= 0 ? i : -1; // 如果净油量大于0，则返回i，否则返回-1
    }

    // 分发糖果
    public int candy(int[] ratings) {
        int n = ratings.length; // 数组长度
        int[] left = new int[n]; // 从左到右遍历
        int[] right = new int[n]; // 从右到左遍历
        Arrays.fill(left, 1); // 初始化left为1
        Arrays.fill(right, 1); // 初始化right为1
        // 从左到右遍历,如果当前位置的评分大于前一个位置的评分,则当前位置的糖果数为前一个位置的糖果数加1
        for (int i = 1; i < n; ++i) { // 遍历数组
            if (ratings[i] > ratings[i - 1]) { // 如果当前位置的评分大于前一个位置的评分
                left[i] = left[i - 1] + 1; // 当前位置的糖果数为前一个位置的糖果数加1
            }
        }
        // 从右到左遍历,如果当前位置的评分大于后一个位置的评分,则当前位置的糖果数为后一个位置的糖果数加1
        for (int i = n - 2; i >= 0; --i) { // 遍历数组
            if (ratings[i] > ratings[i + 1]) { // 如果当前位置的评分大于后一个位置的评分
                right[i] = right[i + 1] + 1; // 当前位置的糖果数为后一个位置的糖果数加1
            }
        }
        // 合并结果，取最大值
        int ans = 0; // 初始化ans为0
        for (int i = 0; i < n; ++i) { // 遍历数组
            ans += Math.max(left[i], right[i]); // 取最大值
        }
        return ans;
    }

    // 接雨水
    public int trap(int[] height) {
        int n = height.length; // 数组长度
        int[] left = new int[n]; // 记录每个位置左侧的最大高度
        int[] right = new int[n]; // 记录每个位置右侧的最大高度
        // 初始化left和right
        left[0] = height[0]; // 初始化left的第一个元素为height的第一个元素
        right[n - 1] = height[n - 1]; // 初始化right的最后一个元素为height的最后一个元素
        // 填充left和right
        for (int i = 1; i < n; ++i) { // 遍历数组
            left[i] = Math.max(left[i - 1], height[i]); // 更新left
            right[n - i - 1] = Math.max(right[n - i], height[n - i - 1]); // 更新right
        }
        // 计算接雨水的量
        int ans = 0; // 初始化ans为0
        for (int i = 0; i < n; ++i) { // 遍历数组
            ans += Math.min(left[i], right[i]) - height[i]; // 计算接雨水的量
        }
        return ans; // 返回结果
    }

    // 罗马数字转整数
    public int romanToInt(String s) {
        // 罗马数字字符串映射表
        String cs = "IVXLCDM"; // 罗马数字字符串
        int[] vs = { 1, 5, 10, 50, 100, 500, 1000 }; // 罗马数字值
        Map<Character, Integer> d = new HashMap<>(); // 存储罗马数字字符及其值
        for (int i = 0; i < cs.length(); ++i) { // 遍历罗马数字字符串
            d.put(cs.charAt(i), vs[i]); // 存储罗马数字字符及其值
        }
        int n = s.length(); // 字符串长度
        int ans = d.get(s.charAt(n - 1)); // 初始化为最后一个字符的值
        // 从第一个字符遍历到倒数第二个字符
        for (int i = 0; i < n - 1; ++i) { // 遍历字符串
            int sign = d.get(s.charAt(i)) < d.get(s.charAt(i + 1)) ? -1 : 1; // 若当前值 <
                                                                             // 下一个值，则减去当前值（如IV中的I），否则加上当前值（如VI中的V）
            // 若当前值 < 下一个值，则减去当前值（如IV中的I），否则加上当前值（如VI中的V）
            ans += sign * d.get(s.charAt(i)); // 更新ans
        }
        return ans; // 返回结果
    }

    // 整数转罗马数字
    public String intToRoman(int num) {
        List<String> cs = List.of("M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"); // 罗马数字字符串
        List<Integer> vs = List.of(1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1); // 罗马数字值
        StringBuilder ans = new StringBuilder(); // 存储结果
        for (int i = 0; i < cs.size(); ++i) { // 遍历罗马数字字符串
            while (num >= vs.get(i)) { // 若num >= 当前值
                ans.append(cs.get(i)); // 添加当前字符
                num -= vs.get(i); // 更新num
            }
        }
        return ans.toString(); // 返回结果
    }

    // 最后一个单词的长度
    public int lengthOfLastWord(String s) {
        int i = s.length() - 1; // 从后往前遍历，找到最后一个单词的结束位置
        // 从后往前遍历，找到最后一个单词的结束位置
        while (i >= 0 && s.charAt(i) == ' ') {
            --i;
        }
        // 从最后一个单词的结束位置往前遍历，找到最后一个单词的起始位置
        int j = i; // 存储最后一个单词的起始位置
        while (j >= 0 && s.charAt(j) != ' ') { // 遍历最后一个单词
            --j; // 减少j
        }
        // i - j即为最后一个单词的长度
        return i - j; // 返回最后一个单词的长度
    }

    // 最长公共前缀
    public String longestCommonPrefix(String[] strs) {
        int n = strs.length; // 字符串数组长度
        for (int i = 0; i < strs[0].length(); ++i) { // 遍历第一个字符串的每个字符
            for (int j = 1; j < n; ++j) { // 遍历其他字符串
                if (strs[j].length() <= i || strs[j].charAt(i) != strs[0].charAt(i)) { // 如果当前字符串的长度小于i，或者当前字符串的第i个字符不等于第一个字符串的第i个字符
                    return strs[0].substring(0, i); // 返回第一个字符串的前i个字符
                }
            }
        }
        return strs[0]; // 如果所有字符串的第i个字符都相等，则返回第一个字符串
    }

    // 反转字符串中的单词
    public String reverseWords(String s) {
        List<String> words = new ArrayList<>(); // 存储单词
        int n = s.length(); // 字符串长度
        for (int i = 0; i < n;) { // 遍历字符串
            // 跳过连续空格
            while (i < n && s.charAt(i) == ' ') {
                ++i; // 增加i
            }
            // 如果i小于n，则说明有单词
            if (i < n) {
                // 获取单词
                StringBuilder t = new StringBuilder(); // 存储单词
                int j = i; // 存储单词的起始位置
                // 找到单词的结束位置
                while (j < n && s.charAt(j) != ' ') {
                    t.append(s.charAt(j++)); // 添加单词字符
                }
                // 将单词加入列表
                words.add(t.toString());
                // 更新i的位置
                i = j;
            }
        }
        // 反转列表
        Collections.reverse(words);
        // 将列表中的单词用空格连接成一个字符串
        return String.join(" ", words);
    }

    // 将字符串转换为Z字形排列
    public String convert(String s, int numRows) {
        if (numRows == 1) {
            return s; // 如果numRows为1，则直接返回s
        }
        StringBuilder[] g = new StringBuilder[numRows]; // 存储字符串
        Arrays.setAll(g, k -> new StringBuilder()); // 初始化g
        int i = 0, k = -1; // 初始化i和k
        for (char c : s.toCharArray()) { // 遍历字符串
            g[i].append(c); // 添加字符
            if (i == 0 || i == numRows - 1) { // 如果i为0或numRows-1
                k = -k; // 更新k
            }
            i += k; // 更新i
        }
        return String.join("", g); // 返回结果
    }

    // 找出字符串中第一个匹配项的下标
    public int strStr(String haystack, String needle) {
        if ("".equals(needle)) {
            return 0; // 如果needle为空，则返回0
        }
        int len1 = haystack.length(); // 字符串长度
        int len2 = needle.length(); // 字符串长度
        int p = 0; // 初始化p
        int q = 0; // 初始化q
        while (p < len1) { // 遍历字符串
            if (haystack.charAt(p) == needle.charAt(q)) { // 如果haystack的第p个字符等于needle的第q个字符
                if (len2 == 1) { // 如果needle的长度为1
                    return p; // 返回p
                }
                ++p; // 增加p
                ++q; // 增加q
            } else { // 如果haystack的第p个字符不等于needle的第q个字符
                p -= q - 1; // 更新p
                q = 0; // 更新q
            }
            if (q == len2) { // 如果q等于needle的长度
                return p - q; // 返回p-q
            }
        }
        return -1; // 如果needle不在haystack中，则返回-1
    }

    // 文本左右对齐
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> ans = new ArrayList<>(); // 存储结果
        for (int i = 0, n = words.length; i < n;) { // 遍历字符串
            List<String> t = new ArrayList<>(); // 存储单词
            t.add(words[i]); // 添加单词
            int cnt = words[i].length(); // 存储单词长度
            ++i; // 增加i
            while (i < n && cnt + 1 + words[i].length() <= maxWidth) { // 遍历字符串
                cnt += 1 + words[i].length(); // 增加cnt
                t.add(words[i++]); // 添加单词
            }
            if (i == n || t.size() == 1) { // 如果i等于n或t的长度为1
                String left = String.join(" ", t); // 存储左边的字符串
                String right = " ".repeat(maxWidth - left.length()); // 存储右边的字符串
                ans.add(left + right); // 添加结果
                continue; // 继续遍历
            }
            int spaceWidth = maxWidth - (cnt - t.size() + 1); // 计算空格宽度
            int w = spaceWidth / (t.size() - 1); // 计算每个单词之间的空格宽度
            int m = spaceWidth % (t.size() - 1); // 计算多余的空格宽度
            StringBuilder row = new StringBuilder(); // 存储结果
            for (int j = 0; j < t.size() - 1; ++j) { // 遍历单词
                row.append(t.get(j)); // 添加单词
                row.append(" ".repeat(w + (j < m ? 1 : 0))); // 添加空格
            }
            row.append(t.get(t.size() - 1)); // 添加最后一个单词
            ans.add(row.toString()); // 添加结果
        }
        return ans; // 返回结果
    }

    // 验证回文串
    public boolean isPalindrome(String s) {
        int i = 0, j = s.length() - 1; // 初始化双指针
        while (i < j) { // 当i小于j时，继续遍历
            if (!Character.isLetterOrDigit(s.charAt(i))) { // 如果i位置不是字母或数字，则i右移
                ++i;
            } else if (!Character.isLetterOrDigit(s.charAt(j))) { // 如果j位置不是字母或数字，则j左移
                --j;
            } else if (Character.toLowerCase(s.charAt(i)) != Character.toLowerCase(s.charAt(j))) {
                return false; // 如果i位置和j位置的字母不相等，则返回false
            } else { // 如果i位置和j位置的字母相等，则i右移，j左移
                ++i; // i右移
                --j; // j左移
            }
        }
        return true; // 如果i和j相遇，则返回true
    }

    // 判断子序列
    public boolean isSubsequence(String s, String t) {
        int m = s.length(), n = t.length(); // 初始化m和n
        int i = 0, j = 0; // 初始化双指针
        while (i < m && j < n) { // 当i小于m且j小于n时，继续遍历
            if (s.charAt(i) == t.charAt(j)) { // 如果s的第i个字符等于t的第j个字符
                ++i; // i右移
            }
            ++j; // 无论i是否等于j，j都右移
        }
        return i == m; // 如果i等于m，则返回true，否则返回false
    }

    // 两数之和 II - 输入有序数组
    public int[] twoSum2(int[] numbers, int target) {
        for (int i = 0, n = numbers.length;; ++i) { // 遍历数组
            int x = target - numbers[i]; // 计算目标值减去当前值
            int l = i + 1, r = n - 1; // 初始化左右指针
            while (l < r) { // 当l小于r时，继续遍历
                int mid = (l + r) >> 1; // 计算中间位置
                if (numbers[mid] >= x) { // 如果numbers[mid]大于等于x
                    r = mid; // 更新r
                } else { // 如果numbers[mid]小于x
                    l = mid + 1; // 更新l
                }
            }
            if (numbers[l] == x) { // 如果numbers[l]等于x
                return new int[] { i + 1, l + 1 }; // 返回结果
            }
        }
    }

    // 盛最多水的容器
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1; // 初始化左右指针
        int ans = 0; // 初始化最大面积
        while (left < right) {
            int t = Math.min(height[left], height[right]) * (right - left); // 计算当前容器的面积
            ans = Math.max(ans, t); // 更新最大面积
            if (height[left] < height[right]) {
                ++left; // 左指针右移
            } else {
                --right; // 右指针左移
            }
        }
        return ans;
    }

    // 三数之和
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums); // 排序
        List<List<Integer>> ans = new ArrayList<>(); // 初始化结果列表
        for (int i = 0; i < n - 2 && nums[i] <= 0; ++i) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue; // 跳过重复元素
            }
            int j = i + 1, k = n - 1; // 初始化双指针
            while (j < k) {
                int x = nums[i] + nums[j] + nums[k]; // 计算三数之和
                if (x < 0) {
                    ++j; // 左指针右移
                } else if (x > 0) {
                    --k; // 右指针左移
                } else {
                    ans.add(List.of(nums[i], nums[j++], nums[k--])); // 添加结果
                    while (j < k && nums[j] == nums[j - 1]) {
                        ++j; // 跳过重复元素
                    }
                    while (j < k && nums[k] == nums[k + 1]) {
                        --k; // 跳过重复元素
                    }
                }
            }
        }
        return ans; // 返回结果列表
    }

    // 最小长度子数组
    public int minSubArrayLen(int target, int[] nums) {
        int n = nums.length; // 数组长度
        long[] s = new long[n + 1]; // 前缀和数组
        for (int i = 0; i < n; ++i) {
            s[i + 1] = s[i] + nums[i]; // 计算前缀和
        }
        int ans = n + 1; // 初始化最小长度
        for (int i = 0; i <= n; ++i) {
            int j = search(s, s[i] + target); // 二分查找
            if (j <= n) {
                ans = Math.min(ans, j - i); // 更新最小长度
            }
        }
        return ans <= n ? ans : 0; // 如果最小长度小于等于n，则返回最小长度，否则返回0
    }

    // 二分查找
    private int search(long[] nums, long x) {
        int left = 0, right = nums.length; // 初始化左右指针
        while (left < right) { // 当left小于right时，继续遍历
            int mid = (left + right) >> 1; // 计算中间位置
            if (nums[mid] >= x) { // 如果nums[mid]大于等于x
                right = mid; // 右指针左移
            } else {
                left = mid + 1; // 左指针右移
            }
        }
        return left; // 返回左指针
    }

    // 无重复字符的最长子串
    public int lengthOfLongestSubstring2(String s) {
        int[] cnt = new int[128]; // 字符计数数组
        int ans = 0, n = s.length(); // 初始化最长子串长度和字符串长度
        for (int left = 0, right = 0; right < n; ++right) { // 遍历字符串
            char c = s.charAt(right); // 获取当前字符
            ++cnt[c]; // 计数加1
            while (cnt[c] > 1) { // 如果当前字符计数大于1
                --cnt[s.charAt(left++)]; // 左指针右移
            }
            ans = Math.max(ans, right - left + 1); // 更新最长子串长度
        }
        return ans; // 返回最长子串长度
    }

    // 串联所有单词的子串
    public List<Integer> findSubstring(String s, String[] words) {
        Map<String, Integer> cnt = new HashMap<>(); // 单词计数
        for (var w : words) {
            cnt.merge(w, 1, Integer::sum); // 计数
        }
        List<Integer> ans = new ArrayList<>(); // 结果列表
        int m = s.length(), n = words.length, k = words[0].length(); // 字符串长度、单词数量、单词长度
        for (int i = 0; i < k; ++i) { // 遍历每个单词
            int left = i, right = i; // 初始化左右指针
            Map<String, Integer> cnt1 = new HashMap<>(); // 当前单词计数
            while (right + k <= m) { // 当右指针加单词长度小于等于字符串长度时
                var t = s.substring(right, right + k); // 获取当前单词
                right += k; // 右指针右移
                if (!cnt.containsKey(t)) { // 如果当前单词不在单词计数中
                    cnt1.clear(); // 清空当前单词计数
                    left = right; // 左指针右移
                    continue; // 跳过当前单词
                }
                cnt1.merge(t, 1, Integer::sum); // 计数
                while (cnt1.get(t) > cnt.get(t)) { // 如果当前单词计数大于单词计数
                    String w = s.substring(left, left + k); // 获取左指针的单词
                    if (cnt1.merge(w, -1, Integer::sum) == 0) { // 如果左指针的单词计数为0
                        cnt1.remove(w); // 移除左指针的单词
                    }
                    left += k; // 左指针右移
                }
                if (right - left == n * k) { // 如果右指针减左指针等于单词数量乘以单词长度
                    ans.add(left); // 添加左指针
                }
            }
        }
        return ans; // 返回结果列表
    }

    // 最小覆盖子串
    public String minWindow(String s, String t) {
        int[] need = new int[128]; // 需要字符数组
        int[] window = new int[128]; // 窗口字符数组
        for (char c : t.toCharArray()) {
            ++need[c]; // 统计t中每个字符出现的次数
        }
        int m = s.length(), n = t.length(); // 字符串长度、t的长度
        int k = -1, mi = m + 1, cnt = 0; // 初始化结果索引、最小长度、计数器
        for (int left = 0, right = 0; right < m; ++right) { // 遍历字符串
            char c = s.charAt(right); // 获取当前字符
            if (++window[c] <= need[c]) { // 如果当前字符出现的次数小于等于t中该字符出现的次数
                ++cnt; // 计数器加1
            }
            while (cnt == n) { // 如果计数器等于t的长度
                if (right - left + 1 < mi) { // 如果当前窗口的长度小于最小长度
                    mi = right - left + 1; // 更新最小长度
                    k = left; // 更新结果索引
                }
                c = s.charAt(left); // 获取左指针的字符
                if (--window[c] <= need[c]) { // 如果左指针的字符出现的次数小于等于t中该字符出现的次数
                    --cnt; // 计数器减1
                }
                --window[c]; // 左指针的字符出现的次数减1
                ++left; // 左指针右移
            }
        }
        return k < 0 ? "" : s.substring(k, k + mi); // 如果k小于0，则返回空字符串，否则返回s的子串
    }

    // 有效的数独
    public boolean isValidSudoku(char[][] board) {
        boolean[][] row = new boolean[9][9]; // 行数组
        boolean[][] col = new boolean[9][9]; // 列数组
        boolean[][] sub = new boolean[9][9]; // 子数组
        for (int i = 0; i < 9; ++i) { // 遍历行
            for (int j = 0; j < 9; ++j) { // 遍历列
                char c = board[i][j]; // 获取当前字符
                if (c == '.') { // 如果当前字符为'.'，则跳过
                    continue;
                }
                int num = c - '0' - 1; // 获取当前字符的数字
                int k = i / 3 * 3 + j / 3; // 获取当前字符的子数组索引
                if (row[i][num] || col[j][num] || sub[k][num]) { // 如果当前字符的行、列、子数组中已经存在该数字，则返回false
                    return false;
                }
                row[i][num] = col[j][num] = sub[k][num] = true; // 将当前字符的行、列、子数组标记为true
            }
        }
        return true; // 返回true
    }

    // 螺旋矩阵
    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length; // 矩阵行数、列数
        int[] dirs = { 0, 1, 0, -1, 0 }; // 方向数组
        int i = 0, j = 0, k = 0; // 初始化行、列、方向
        List<Integer> ans = new ArrayList<>(); // 结果列表
        boolean[][] vis = new boolean[m][n]; // 访问数组
        for (int h = m * n; h > 0; --h) { // 遍历矩阵
            ans.add(matrix[i][j]); // 添加当前元素
            vis[i][j] = true; // 标记为访问
            int x = i + dirs[k], y = j + dirs[k + 1]; // 计算下一个元素的行、列
            if (x < 0 || x >= m || y < 0 || y >= n || vis[x][y]) { // 如果下一个元素的行、列超出范围或已访问，则改变方向
                k = (k + 1) % 4; // 改变方向
            }
            i += dirs[k]; // 更新行
            j += dirs[k + 1]; // 更新列
        }
        return ans; // 返回结果列表
    }

    // 旋转图像
    public void rotate(int[][] matrix) {
        int n = matrix.length; // 矩阵行数
        for (int i = 0; i < n >> 1; ++i) { // 遍历矩阵
            for (int j = 0; j < n; ++j) { // 遍历矩阵
                int t = matrix[i][j]; // 交换矩阵元素
                matrix[i][j] = matrix[n - i - 1][j]; // 交换矩阵元素
                matrix[n - i - 1][j] = t; // 交换矩阵元素
            }
        }
        for (int i = 0; i < n; ++i) { // 遍历矩阵
            for (int j = 0; j < i; ++j) { // 遍历矩阵
                int t = matrix[i][j]; // 交换矩阵元素
                matrix[i][j] = matrix[j][i]; // 交换矩阵元素
                matrix[j][i] = t; // 交换矩阵元素
            }
        }
    }

    // 矩阵置零
    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length; // 矩阵行数、列数
        boolean[] row = new boolean[m]; // 行数组
        boolean[] col = new boolean[n]; // 列数组
        for (int i = 0; i < m; ++i) { // 遍历矩阵
            for (int j = 0; j < n; ++j) { // 遍历矩阵
                if (matrix[i][j] == 0) { // 如果当前元素为0
                    row[i] = col[j] = true; // 将行和列标记为true
                }
            }
        }
        for (int i = 0; i < m; ++i) { // 遍历矩阵
            for (int j = 0; j < n; ++j) { // 遍历矩阵
                if (row[i] || col[j]) { // 如果当前行或列被标记为true
                    matrix[i][j] = 0; // 将当前元素置为0
                }
            }
        }
    }

    // 生命游戏
    public void gameOfLife(int[][] board) {
        int m = board.length, n = board[0].length; // 矩阵行数、列数
        for (int i = 0; i < m; ++i) { // 遍历矩阵
            for (int j = 0; j < n; ++j) { // 遍历矩阵
                int live = -board[i][j]; // 当前细胞状态
                for (int x = i - 1; x <= i + 1; ++x) { // 遍历周围细胞
                    for (int y = j - 1; y <= j + 1; ++y) { // 遍历周围细胞
                        if (x >= 0 && x < m && y >= 0 && y < n && board[x][y] > 0) { // 如果周围细胞存在且为活细胞
                            ++live; // 活细胞数量加1
                        }
                    }
                }
                if (board[i][j] == 1 && (live < 2 || live > 3)) { // 如果当前细胞为活细胞且周围活细胞数量小于2或大于3
                    board[i][j] = 2; // 当前细胞状态变为2
                }
                if (board[i][j] == 0 && live == 3) { // 如果当前细胞为死细胞且周围活细胞数量为3
                    board[i][j] = -1; // 当前细胞状态变为-1
                }
            }
        }
        for (int i = 0; i < m; ++i) { // 遍历矩阵
            for (int j = 0; j < n; ++j) { // 遍历矩阵
                if (board[i][j] == 2) { // 如果当前细胞状态为2
                    board[i][j] = 0; // 当前细胞状态变为0
                } else if (board[i][j] == -1) { // 如果当前细胞状态为-1
                    board[i][j] = 1; // 当前细胞状态变为1
                }
            }
        }
    }

    // 赎金信
    public boolean canConstruct(String ransomNote, String magazine) {
        int[] cnt = new int[26]; // 字符计数数组
        for (int i = 0; i < magazine.length(); ++i) { // 遍历magazine
            ++cnt[magazine.charAt(i) - 'a']; // 统计magazine中每个字符出现的次数
        }
        for (int i = 0; i < ransomNote.length(); ++i) { // 遍历ransomNote
            if (--cnt[ransomNote.charAt(i) - 'a'] < 0) { // 如果当前字符出现的次数小于0
                return false; // 返回false
            }
        }
        return true; // 返回true
    }

    // 同构字符串
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> d1 = new HashMap<>(); // 映射表1
        Map<Character, Character> d2 = new HashMap<>(); // 映射表2
        int n = s.length(); // 字符串长度
        for (int i = 0; i < n; ++i) { // 遍历字符串
            char a = s.charAt(i), b = t.charAt(i); // 获取当前字符
            if (d1.containsKey(a) && d1.get(a) != b || d2.containsKey(b) && d2.get(b) != a) { // 如果当前字符的映射表1或映射表2中已经存在该字符，则返回false
                return false;
            }
            d1.put(a, b); // 将当前字符的映射表1映射为当前字符
            d2.put(b, a); // 将当前字符的映射表2映射为当前字符
        }
        return true; // 返回true
    }

    // 单词规律
    public boolean wordPattern(String pattern, String s) {
        String[] words = s.split(" "); // 将字符串s分割成单词数组
        if (pattern.length() != words.length) { // 如果pattern的长度不等于单词数组的长度，则返回false
            return false;
        }
        Map<Character, String> d1 = new HashMap<>(); // 映射表1
        Map<String, Character> d2 = new HashMap<>(); // 映射表2
        for (int i = 0; i < words.length; ++i) { // 遍历字符串
            char a = pattern.charAt(i); // 获取当前字符
            String b = words[i]; // 获取当前单词
            if (!d1.getOrDefault(a, b).equals(b) || d2.getOrDefault(b, a) != a) { // 如果当前字符的映射表1或映射表2中已经存在该字符，则返回false
                return false;
            }
            d1.put(a, b); // 将当前字符的映射表1映射为当前字符
            d2.put(b, a); // 将当前字符的映射表2映射为当前字符
        }
        return true; // 返回true
    }

    // 有效的字母异位词
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) { // 如果s的长度不等于t的长度，则返回false
            return false;
        }
        int[] cnt = new int[26]; // 字符计数数组
        for (int i = 0; i < s.length(); ++i) { // 遍历字符串
            ++cnt[s.charAt(i) - 'a']; // 统计s中每个字符出现的次数
            --cnt[t.charAt(i) - 'a']; // 统计t中每个字符出现的次数
        }
        for (int i = 0; i < 26; ++i) { // 遍历字符计数数组
            if (cnt[i] != 0) { // 如果当前字符出现的次数不为0，则返回false
                return false;
            }
        }
        return true; // 返回true
    }

    // 字母异位词分组
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> d = new HashMap<>(); // 映射表
        for (String s : strs) { // 遍历字符串数组
            char[] cs = s.toCharArray(); // 将字符串转换为字符数组
            Arrays.sort(cs); // 对字符数组进行排序
            String k = String.valueOf(cs); // 将字符数组转换为字符串
            d.computeIfAbsent(k, key -> new ArrayList<>()).add(s); // 将字符串添加到映射表中
        }
        return new ArrayList<>(d.values()); // 返回映射表的值
    }

    // 两数之和
    public int[] twoSum1(int[] nums, int target) {
        Map<Integer, Integer> d = new HashMap<>(); // 映射表
        for (int i = 0;; ++i) { // 遍历数组
            int x = nums[i]; // 获取当前元素
            int y = target - x; // 获取目标值减去当前元素
            if (d.containsKey(y)) { // 如果映射表中存在该元素
                return new int[] { d.get(y), i }; // 返回该元素的索引
            }
            d.put(x, i); // 将当前元素的索引添加到映射表中
        }
    }

    // 快乐数
    public boolean isHappy(int n) {
        Set<Integer> vis = new HashSet<>(); // 访问集合
        while (n != 1 && !vis.contains(n)) { // 如果n不等于1且未访问过n
            vis.add(n); // 将n添加到访问集合中
            int x = 0; // 初始化x
            while (n != 0) { // 如果n不等于0
                x += (n % 10) * (n % 10); // 计算n的平方和
                n /= 10; // 将n除以10
            }
            n = x; // 更新n
        }
        return n == 1; // 如果n等于1，则返回true，否则返回false
    }

    // 存在重复元素
    public boolean containsDuplicate(int[] nums) {
        Arrays.sort(nums); // 对数组进行排序
        for (int i = 0; i < nums.length - 1; ++i) { // 遍历数组
            if (nums[i] == nums[i + 1]) { // 如果当前元素等于下一个元素
                return true; // 返回true
            }
        }
        return false; // 返回false
    }

    // 存在重复元素 II
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> d = new HashMap<>(); // 映射表
        for (int i = 0; i < nums.length; ++i) { // 遍历数组
            if (i - d.getOrDefault(nums[i], -1000000) <= k) { // 如果当前元素的索引减去映射表中该元素的索引小于等于k
                return true; // 返回true
            }
            d.put(nums[i], i); // 将当前元素的索引添加到映射表中
        }
        return false; // 返回false
    }

    // 最长连续序列
    public int longestConsecutive(int[] nums) {
        Set<Integer> s = new HashSet<>(); // 集合
        for (int x : nums) { // 遍历数组
            s.add(x); // 将当前元素添加到集合中
        }
        int ans = 0; // 结果
        Map<Integer, Integer> d = new HashMap<>(); // 映射表
        for (int x : nums) { // 遍历数组
            int y = x; // 当前元素
            while (s.contains(y)) { // 如果集合中存在当前元素
                s.remove(y++); // 更新当前元素
            }
            d.put(x, d.getOrDefault(y, 0) + y - x); // 将当前元素的映射表值添加到映射表中
            ans = Math.max(ans, d.get(x)); // 更新结果
        }
        return ans; // 返回结果
    }

    // 汇总区间
    public List<String> summaryRanges(int[] nums) {
        List<String> ans = new ArrayList<>(); // 结果列表
        for (int i = 0, j, n = nums.length; i < n; i = j + 1) { // 遍历数组
            j = i; // 当前元素
            while (j + 1 < n && nums[j + 1] == nums[j] + 1) { // 如果当前元素加1等于下一个元素
                ++j; // 更新当前元素
            }
            ans.add(f(nums, i, j)); // 将当前元素添加到结果列表中
        }
        return ans; // 返回结果列表
    }

    // 辅助函数
    private String f(int[] nums, int i, int j) {
        return i == j ? nums[i] + "" : String.format("%d->%d", nums[i], nums[j]); // 返回结果
    }

    // 合并区间
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0])); // 对数组进行排序
        int st = intervals[0][0], ed = intervals[0][1]; // 初始化开始和结束
        List<int[]> ans = new ArrayList<>(); // 结果列表
        for (int i = 1; i < intervals.length; ++i) { // 遍历数组
            int s = intervals[i][0], e = intervals[i][1]; // 获取当前元素的开始和结束
            if (s > ed) { // 如果当前元素的开始大于结束
                ans.add(new int[] { st, ed }); // 将当前元素添加到结果列表中
                st = s; // 更新开始
                ed = e; // 更新结束
            } else {
                ed = Math.max(ed, e); // 更新结束
            }
        }
        ans.add(new int[] { st, ed }); // 将当前元素添加到结果列表中
        return ans.toArray(new int[ans.size()][]); // 返回结果列表
    }

    // 插入区间
    public int[][] insert(int[][] intervals, int[] newInterval) {
        int[][] newIntervals = new int[intervals.length + 1][2]; // 新的区间数组
        for (int i = 0; i < intervals.length; ++i) { // 遍历区间数组
            newIntervals[i] = intervals[i]; // 将当前元素添加到新的区间数组中
        }
        newIntervals[intervals.length] = newInterval; // 将新的区间添加到新的区间数组中
        return merge2(newIntervals); // 返回合并后的区间数组
    }

    // 合并区间
    private int[][] merge2(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]); // 对数组进行排序
        List<int[]> ans = new ArrayList<>(); // 结果列表
        ans.add(intervals[0]); // 将当前元素添加到结果列表中
        for (int i = 1; i < intervals.length; ++i) { // 遍历区间数组
            int s = intervals[i][0], e = intervals[i][1]; // 获取当前元素的开始和结束
            if (ans.get(ans.size() - 1)[1] < s) { // 如果结果列表的最后一个元素的结束小于当前元素的开始
                ans.add(intervals[i]); // 将当前元素添加到结果列表中
            } else {
                ans.get(ans.size() - 1)[1] = Math.max(ans.get(ans.size() - 1)[1], e); // 更新结果列表的最后一个元素的结束
            }
        }
        return ans.toArray(new int[ans.size()][]); // 返回结果列表
    }

    // 用最少数量的箭引爆气球
    public int findMinArrowShots(int[][] points) {
        Arrays.sort(points, Comparator.comparingInt(a -> a[1])); // 对数组进行排序
        int ans = 0; // 结果
        long last = -(1L << 60); // 最后一个箭射的位置
        for (var p : points) { // 遍历数组
            int a = p[0], b = p[1]; // 获取当前元素的开始和结束
            if (a > last) {
                ++ans; // 增加结果
                last = b; // 更新最后一个箭射的位置
            }
        }
        return ans; // 返回结果
    }

    // 有效的括号
    public boolean isValid(String s) {
        Deque<Character> stk = new ArrayDeque<>(); // 双端队列
        for (char c : s.toCharArray()) { // 遍历字符串
            if (c == '(' || c == '[' || c == '{') { // 如果当前字符是左括号
                stk.push(c); // 将当前字符添加到双端队列中
            } else if (stk.isEmpty() || !match(stk.pop(), c)) { // 如果双端队列为空或当前字符不等于双端队列的最后一个元素
                return false; // 返回false
            }
        }
        return stk.isEmpty(); // 返回双端队列是否为空
    }

    // 辅助函数
    private boolean match(char left, char right) {
        return (left == '(' && right == ')') || (left == '[' && right == ']') || (left == '{' && right == '}'); // 返回是否匹配
    }

    // 简化路径
    public String simplifyPath(String path) {
        Deque<String> stk = new ArrayDeque<>(); // 双端队列
        for (String s : path.split("/")) { // 遍历字符串
            if ("".equals(s) || ".".equals(s)) { // 如果当前字符串是空或.
                continue; // 跳过
            }
            if ("..".equals(s)) { // 如果当前字符串是..
                stk.pollLast(); // 弹出双端队列的最后一个元素
            } else {
                stk.offerLast(s); // 将当前字符串添加到双端队列中
            }
        }
        return "/" + String.join("/", stk); // 返回结果
    }

    // 最小栈
    class MinStack {
        private Deque<Integer> stk1 = new ArrayDeque<>(); // 栈1
        private Deque<Integer> stk2 = new ArrayDeque<>(); // 栈2

        public MinStack() { // 构造函数
            stk2.push(Integer.MAX_VALUE); // 初始化栈2
        }

        public void push(int val) { // 入栈
            stk1.push(val); // 入栈1
            stk2.push(Math.min(stk2.peek(), val)); // 入栈2
        }

        public void pop() { // 出栈
            stk1.pop(); // 出栈1
            stk2.pop(); // 出栈2
        }

        public int top() { // 获取栈顶元素
            return stk1.peek(); // 返回栈1的栈顶元素
        }

        public int getMin() { // 获取最小值
            return stk2.peek(); // 返回栈2的栈顶元素
        }
    }

    // 逆波兰表达式求值
    public int evalRPN(String[] tokens) {
        Deque<Integer> stk = new ArrayDeque<>(); // 双端队列
        for (String t : tokens) { // 遍历字符串数组
            if (t.length() > 1 || Character.isDigit(t.charAt(0))) { // 如果当前字符串的长度大于1或当前字符串的第一个字符是数字
                stk.push(Integer.parseInt(t)); // 将当前字符串转换为整数并入栈
            } else {
                int x = stk.pop(), y = stk.pop(); // 获取栈顶的两个元素
                switch (t) { // 根据当前字符串进行操作
                    case "+":
                        stk.push(x + y); // 将当前元素入栈
                        break;
                    case "-":
                        stk.push(x - y); // 将当前元素入栈
                        break;
                    case "*":
                        stk.push(x * y); // 将当前元素入栈
                        break;
                    default:
                        stk.push(x / y); // 将当前元素入栈
                        break;
                }
            }
        }
        return stk.pop(); // 返回栈顶元素
    }

    // 基本计算器
    public int calculate(String s) {
        Deque<Integer> stk = new ArrayDeque<>(); // 双端队列
        int sign = 1; // 符号
        int ans = 0; // 结果
        int n = s.length(); // 字符串长度
        for (int i = 0; i < n; ++i) { // 遍历字符串
            char c = s.charAt(i); // 获取当前字符
            if (Character.isDigit(c)) { // 如果当前字符是数字
                int j = i; // 当前字符索引
                int x = 0; // 当前数字
                while (j < n && Character.isDigit(s.charAt(j))) { // 如果当前字符是数字
                    x = x * 10 + s.charAt(j) - '0'; // 将当前字符转换为数字
                    j++; // 更新当前字符索引
                }
                ans += sign * x; // 更新结果
                i = j - 1; // 更新当前字符索引
            } else if (c == '+') { // 如果当前字符是+
                sign = 1; // 更新符号
            } else if (c == '-') { // 如果当前字符是-
                sign = -1; // 更新符号
            } else if (c == '(') { // 如果当前字符是(
                stk.push(ans); // 将当前结果入栈
                stk.push(sign); // 将当前符号入栈
                ans = 0; // 更新结果
                sign = 1; // 更新符号
            } else if (c == ')') { // 如果当前字符是)
                ans = stk.pop() * ans + stk.pop(); // 更新结果
            }
        }
        return ans; // 返回结果
    }

    // 环形链表
    public boolean hasCycle(ListNode head) {
        Set<ListNode> s = new HashSet<>(); // 访问集合
        for (; head != null; head = head.next) { // 遍历链表
            if (!s.add(head)) { // 如果当前节点已经访问过，则返回true
                return true; // 返回true
            }
        }
        return false; // 返回false
    }

    // 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0); // 虚拟头节点
        int carry = 0; // 进位
        ListNode cur = dummy; // 当前节点
        while (l1 != null || l2 != null || carry != 0) { // 遍历链表
            int s = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + carry; // 计算当前节点的值
            carry = s / 10; // 更新进位
            cur.next = new ListNode(s % 10); // 更新当前节点的值
            cur = cur.next; // 更新当前节点
            l1 = l1 == null ? null : l1.next; // 更新l1
            l2 = l2 == null ? null : l2.next; // 更新l2
        }
        return dummy.next; // 返回虚拟头节点的下一个节点
    }

    // 合并两个有序链表
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) { // 如果list1为空
            return list2; // 返回list2
        }
        if (list2 == null) { // 如果list2为空
            return list1; // 返回list1
        }
        if (list1.val <= list2.val) { // 如果list1的值小于等于list2的值
            list1.next = mergeTwoLists(list1.next, list2); // 递归调用mergeTwoLists
            return list1; // 返回list1
        } else {
            list2.next = mergeTwoLists(list1, list2.next); // 递归调用mergeTwoLists
            return list2; // 返回list2
        }
    }

    // Add Node class definition
    static class Node {
        int val; // 值
        Node next; // 下一个节点
        Node random; // 随机节点

        public Node(int val) {
            this.val = val; // 初始化值
            this.next = null; // 初始化下一个节点
            this.random = null; // 初始化随机节点
        }
    }

    class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    // 复制带随机指针的链表
    public Node copyRandomList(Node head) {
        Map<Node, Node> d = new HashMap<>(); // 映射表
        Node dummy = new Node(0); // 虚拟头节点
        Node tail = dummy; // 尾节点
        for (Node cur = head; cur != null; cur = cur.next) { // 遍历链表
            Node node = new Node(cur.val); // 创建新节点
            tail.next = node; // 将新节点添加到尾节点
            tail = node; // 更新尾节点
            d.put(cur, node); // 将当前节点添加到映射表中
        }
        for (Node cur = head; cur != null; cur = cur.next) { // 遍历链表
            d.get(cur).random = cur.random == null ? null : d.get(cur.random); // 更新随机指针
        }
        return dummy.next; // 返回虚拟头节点的下一个节点
    }

    // 反转链表 II
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (head.next == null || left == right) { // 如果链表为空或left等于right
            return head; // 返回head
        }
        ListNode dummy = new ListNode(0, head); // 虚拟头节点
        ListNode pre = dummy; // 前一个节点
        for (int i = 1; i < left - 1; ++i) { // 遍历链表
            pre = pre.next; // 更新前一个节点
        }
        ListNode p = pre; // 前一个节点
        ListNode q = pre.next; // 当前节点
        ListNode cur = q; // 当前节点
        for (int i = 0; i < right - left + 1; ++i) { // 遍历链表
            ListNode t = cur.next; // 下一个节点
            cur.next = pre; // 更新当前节点的下一个节点
            pre = cur; // 更新前一个节点
            cur = t; // 更新当前节点
        }
        p.next = pre; // 更新前一个节点的下一个节点
        q.next = cur; // 更新当前节点的下一个节点
        return dummy.next; // 返回虚拟头节点的下一个节点
    }

    // K 个一组翻转链表
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0, head); // 虚拟头节点
        dummy.next = head; // 虚拟头节点的下一个节点
        ListNode pre = dummy; // 前一个节点
        while (pre != null) { // 遍历链表
            ListNode cur = pre; // 当前节点
            for (int i = 0; i < k; i++) { // 遍历链表
                cur = cur.next; // 更新当前节点
                if (cur == null) { // 如果当前节点为空
                    return dummy.next; // 返回虚拟头节点的下一个节点
                }
            }
            ListNode node = pre.next; // 下一个节点
            ListNode nxt = cur.next; // 下一个节点
            cur.next = null; // 更新当前节点的下一个节点
            pre.next = reverse(node); // 更新前一个节点的下一个节点
            node.next = nxt; // 更新下一个节点的下一个节点
            pre = node; // 更新前一个节点
        }
        return dummy.next; // 返回虚拟头节点的下一个节点
    }

    // 辅助函数
    private ListNode reverse(ListNode head) {
        ListNode dummy = new ListNode(); // 虚拟头节点
        ListNode cur = head; // 当前节点
        while (cur != null) { // 遍历链表
            ListNode nxt = cur.next; // 下一个节点
            cur.next = dummy.next; // 更新当前节点的下一个节点
            dummy.next = cur; // 更新虚拟头节点的下一个节点
            cur = nxt; // 更新当前节点
        }
        return dummy.next; // 返回虚拟头节点的下一个节点
    }

    // 删除链表的倒数第 N 个结点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head); // 虚拟头节点
        ListNode fast = dummy; // 快指针
        ListNode slow = dummy; // 慢指针
        while (n-- > 0) { // 遍历链表
            fast = fast.next; // 更新快指针
        }
        while (fast.next != null) { // 遍历链表
            fast = fast.next; // 更新快指针
            slow = slow.next; // 更新慢指针
        }
        slow.next = slow.next.next; // 更新慢指针的下一个节点
        return dummy.next; // 返回虚拟头节点的下一个节点
    }

    // 删除排序链表中的重复元素
    public ListNode deleteDuplicates(ListNode head) {
        ListNode dummy = new ListNode(0, head); // 虚拟头节点
        ListNode pre = dummy; // 前一个节点
        ListNode cur = head; // 当前节点
        while (cur != null) { // 遍历链表
            while (cur.next != null && cur.val == cur.next.val) { // 如果当前节点的下一个节点不为空且当前节点的值等于下一个节点的值
                cur = cur.next; // 更新当前节点
            }
            if (pre.next == cur) { // 如果前一个节点的下一个节点等于当前节点
                pre = cur; // 更新前一个节点
            } else {
                pre.next = cur.next; // 更新前一个节点的下一个节点
            }
            cur = cur.next; // 更新当前节点
        }
        return dummy.next; // 返回虚拟头节点的下一个节点
    }

    // 旋转链表
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || head.next == null) { // 如果链表为空或只有一个节点
            return head; // 返回head
        }
        ListNode cur = head; // 当前节点
        int n = 0; // 链表长度
        for (; cur != null; cur = cur.next) { // 遍历链表
            n++; // 更新链表长度
        }
        k = k % n; // 更新k
        if (k == 0) { // 如果k为0
            return head; // 返回head
        }
        ListNode fast = head; // 快指针
        ListNode slow = head; // 慢指针
        while (k-- > 0) { // 遍历链表
            fast = fast.next; // 更新快指针
        }
        while (fast.next != null) { // 遍历链表
            fast = fast.next; // 更新快指针
            slow = slow.next; // 更新慢指针
        }
        ListNode ans = slow.next; // 新头节点
        slow.next = null; // 断开链表
        fast.next = head; // 连接链表
        return ans; // 返回新头节点
    }

    // 分割链表
    public ListNode[] splitListToParts(ListNode head, int k) {
        int n = 0; // 链表长度
        for (ListNode cur = head; cur != null; cur = cur.next) { // 遍历链表
            ++n; // 更新链表长度
        }
        int cnt = n / k, mod = n % k; // 计算每个链表的长度
        ListNode[] ans = new ListNode[k]; // 创建结果数组
        ListNode cur = head; // 当前节点
        for (int i = 0; i < k && cur != null; ++i) { // 遍历链表
            ans[i] = cur; // 将当前节点赋值给结果数组
            int m = cnt + (i < mod ? 1 : 0); // 计算每个链表的长度
            for (int j = 1; j < m; ++j) { // 遍历链表
                cur = cur.next; // 更新当前节点
            }
            ListNode next = cur.next; // 下一个节点
            cur.next = null; // 断开链表
            cur = next; // 更新当前节点
        }
        return ans; // 返回结果数组
    }
}

// 双向链表
class Node {
    int key, val; // 键值对
    Node prev, next; // 前驱和后继

    Node() { // 构造函数
    }

    Node(int key, int val) { // 构造函数
        this.key = key; // 初始化键
        this.val = val; // 初始化值
    }
}

class LRUCache { // 双向链表
    private int size; // 大小
    private int capacity; // 容量
    private Map<Integer, Node> cache = new HashMap<>(); // 缓存
    private Node head = new Node(); // 头节点
    private Node tail = new Node(); // 尾节点

    public LRUCache(int capacity) { // 构造函数
        this.capacity = capacity; // 初始化容量
        head.next = tail; // 头节点的下一个节点
        tail.prev = head; // 尾节点的前一个节点
    }

    // 辅助函数
    private void addToHead(Node node) {
        node.next = head.next; // 将节点添加到头节点之后
        node.prev = head; // 将节点的前一个节点设置为头节点
        head.next.prev = node; // 将头节点的下一个节点的前一个节点设置为节点
        head.next = node; // 将头节点的下一个节点设置为节点
    }

    private void removeNode(Node node) {
        node.prev.next = node.next; // 将节点的前一个节点的下一个节点设置为节点的下一个节点
        node.next.prev = node.prev; // 将节点的下一个节点的前一个节点设置为节点的前一个节点
    }

    public int get(int key) {
        if (!cache.containsKey(key)) { // 如果缓存中不包含该键
            return -1; // 返回-1
        }
        Node node = cache.get(key); // 获取节点
        removeNode(node); // 移除节点
        addToHead(node); // 添加到头节点
        return node.val; // 返回节点的值
    }

    public void put(int key, int value) { // 添加节点
        if (cache.containsKey(key)) { // 如果缓存中包含该键
            Node node = cache.get(key); // 获取节点
            removeNode(node); // 移除节点
            node.val = value; // 更新节点的值
            addToHead(node); // 添加到头节点
        } else {
            Node node = new Node(key, value); // 创建新节点
            cache.put(key, node); // 添加到缓存
            addToHead(node); // 添加到头节点
            if (++size > capacity) { // 如果大于容量
                node = tail.prev; // 获取尾节点
                cache.remove(node.key); // 移除节点
                removeNode(node); // 移除节点
                --size; // 更新大小
            }
        }
    }
}

class Solution2 {
    // 二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) { // 如果根节点为空
            return 0; // 返回0
        }
        int leftDepth = maxDepth(root.left); // 左子树的最大深度
        int rightDepth = maxDepth(root.right); // 右子树的最大深度
        return 1 + Math.max(leftDepth, rightDepth); // 返回1加上左右子树的最大深度
    }

    // 二叉树节点(二叉树模块通用类)
    class TreeNode {
        int val; // 节点值
        TreeNode left; // 左子树
        TreeNode right; // 右子树

        TreeNode() { // 构造函数
        }

        TreeNode(int val) { // 构造函数
            this.val = val; // 初始化节点值
        }

        TreeNode(int val, TreeNode left, TreeNode right) { // 构造函数
            this.val = val; // 初始化节点值
            this.left = left; // 初始化左子树
            this.right = right; // 初始化右子树
        }
    }

    // 相同的树
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == q)
            return true; // 如果两个节点相同
        if (p == null || q == null || p.val != q.val)
            return false; // 如果两个节点不同
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right); // 递归判断左右子树
    }

    // 翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        if (root == null) { // 如果根节点为空
            return null; // 返回null
        }
        TreeNode left = invertTree(root.left); // 翻转左子树
        TreeNode right = invertTree(root.right); // 翻转右子树
        root.left = right; // 交换左右子树
        root.right = left; // 交换左右子树
        return root; // 返回根节点
    }

    public boolean isSymmetric(TreeNode root) {
        return dfs(root.left, root.right);
    }

    // 对称二叉树
    private boolean dfs(TreeNode root1, TreeNode root2) {
        if (root1 == root2) { // 如果两个节点相同
            return true; // 返回true
        }
        if (root1 == null || root2 == null || root1.val != root2.val) { // 如果两个节点不同
            return false; // 返回false
        }
        return dfs(root1.left, root2.right) && dfs(root1.right, root2.left); // 递归判断左右子树
    }

    // 从前序与中序遍历序列构造二叉树
    private int[] preorder; // 前序遍历序列
    private Map<Integer, Integer> d = new HashMap<>(); // 中序遍历序列

    public TreeNode buildTree(int[] preorder, int[] inorder) { // 构造二叉树
        int n = preorder.length; // 前序遍历序列的长度
        this.preorder = preorder; // 初始化前序遍历序列
        for (int i = 0; i < n; ++i) { // 遍历中序遍历序列
            d.put(inorder[i], i); // 将中序遍历序列的值和索引添加到映射表中
        }
        return dfs(0, 0, n); // 递归构造二叉树
    }

    private TreeNode dfs(int i, int j, int n) { // 辅助函数
        if (n <= 0) { // 如果n小于等于0
            return null; // 返回null
        }
        int v = preorder[i]; // 获取前序遍历序列的值
        int k = d.get(v); // 获取中序遍历序列的值
        TreeNode left = dfs(i + 1, j, k - j); // 递归构造左子树
        TreeNode right = dfs(i + k - j + 1, k + 1, n - (k - j + 1)); // 递归构造右子树
        return new TreeNode(v, left, right); // 返回新节点
    }

    // 填充每个节点的下一个右侧节点指针 II使用类
    class Node {
        public int val; // 节点值
        public Node left; // 左子树
        public Node right; // 右子树
        public Node next; // 下一个节点

        public Node() { // 构造函数
        }

        public Node(int _val) { // 构造函数
            this.val = _val; // 初始化节点值
        }

        public Node(int _val, Node _left, Node _right, Node _next) { // 构造函数
            this.val = _val; // 初始化节点值
            this.left = _left; // 初始化左子树
            this.right = _right; // 初始化右子树
            this.next = _next; // 初始化下一个节点
        }
    }

    // 填充每个节点的下一个右侧节点指针 II
    public Node connect(Node root) {
        if (root == null) { // 如果根节点为空
            return root; // 返回根节点
        }
        Deque<Node> q = new ArrayDeque<>(); // 创建队列
        q.offer(root); // 添加根节点
        while (!q.isEmpty()) { // 如果队列不为空
            Node p = null; // 当前节点
            for (int n = q.size(); n > 0; --n) { // 遍历队列
                Node node = q.poll(); // 获取队列中的节点
                if (p != null) { // 如果当前节点不为空
                    p.next = node; // 更新当前节点的下一个节点
                }
                p = node; // 更新当前节点
                if (node.left != null) { // 如果左子树不为空
                    q.offer(node.left); // 添加左子树
                }
                if (node.right != null) { // 如果右子树不为空
                    q.offer(node.right); // 添加右子树
                }
            }
        }
        return root; // 返回根节点
    }

    // 二叉树展开为链表
    public void flatten(TreeNode root) {
        while (root != null) { // 如果根节点不为空
            if (root.left != null) { // 如果左子树不为空
                TreeNode pre = root.left; // 前一个节点
                while (pre.right != null) { // 如果前一个节点的右子树不为空
                    pre = pre.right; // 更新前一个节点
                }
                pre.right = root.right; // 更新前一个节点的右子树
                root.right = root.left; // 更新根节点的右子树
                root.left = null; // 更新根节点的左子树
            }
            root = root.right; // 更新根节点
        }
    }

    // 路径总和
    public boolean hasPathSum(TreeNode root, int targetSum) { // 判断是否存在路径总和等于targetSum
        return dfs(root, targetSum); // 递归判断
    }

    private boolean dfs(TreeNode root, int s) {
        if (root == null) { // 如果根节点为空
            return false; // 返回false
        }
        s -= root.val;
        if (root.left == null && root.right == null && s == 0) { // 如果左右子树为空且s等于0
            return true; // 返回true
        }
        return dfs(root.left, s) || dfs(root.right, s); // 递归判断左右子树
    }

    // 求根节点到叶节点数字之和
    public int sumNumbers(TreeNode root) {
        return dfs2(root, 0); // 递归判断
    }

    // 辅助函数
    private int dfs2(TreeNode root, int s) {
        if (root == null) { // 如果根节点为空
            return 0; // 返回0
        }
        s = s * 10 + root.val; // 更新s
        if (root.left == null && root.right == null) { // 如果左右子树为空
            return s; // 返回s
        }
        return dfs2(root.left, s) + dfs2(root.right, s); // 递归判断左右子树
    }

    // 二叉树中的最大路径和
    private int ans = -1001; // 最大路径和

    public int maxPathSum(TreeNode root) { // 最大路径和
        dfs3(root); // 递归判断
        return ans; // 返回最大路径和
    }

    // 辅助函数
    private int dfs3(TreeNode root) {
        if (root == null) { // 如果根节点为空
            return 0; // 返回0
        }
        int left = Math.max(0, dfs3(root.left)); // 左子树的最大路径和
        int right = Math.max(0, dfs3(root.right)); // 右子树的最大路径和
        ans = Math.max(ans, root.val + left + right); // 更新最大路径和
        return root.val + Math.max(left, right); // 返回根节点的值加上左右子树的最大路径和
    }

    // 二叉搜索树迭代器
    class BSTIterator {
        private int cur = 0; // 当前索引
        private List<Integer> vals = new ArrayList<>(); // 存储中序遍历的值

        public BSTIterator(TreeNode root) { // 构造函数
            inorder(root); // 中序遍历
        }

        public int next() { // 获取下一个值
            return vals.get(cur++); // 返回当前索引的值
        }

        public boolean hasNext() { // 判断是否存在下一个值
            return cur < vals.size(); // 返回是否存在下一个值
        }

        private void inorder(TreeNode root) { // 中序遍历
            if (root == null) { // 如果根节点为null
                inorder(root.left); // 中序遍历左子树
                vals.add(root.val); // 添加根节点的值
                inorder(root.right); // 中序遍历右子树
            }
        }
    }

    // 完全二叉树的节点个数
    public int countNodes(TreeNode root) {
        if (root == null) { // 如果根节点为空
            return 0; // 返回0
        }
        return 1 + countNodes(root.left) + countNodes(root.right); // 返回1加上左右子树的节点个数
    }

    // 二叉树的最近公共祖先
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) { // 如果根节点为空或根节点为p或根节点为q
            return root; // 返回根节点
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q); // 递归判断左子树
        TreeNode right = lowestCommonAncestor(root.right, p, q); // 递归判断右子树
        if (left != null && right != null) { // 如果左子树和右子树都不为空
            return root; // 返回根节点
        }
        return left != null ? left : right; // 返回左子树或右子树
    }

    // 二叉树的右视图
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> ans = new ArrayList<>(); // 存储右视图的值
        if (root == null) { // 如果根节点为空
            return ans; // 返回空列表
        }
        Deque<TreeNode> q = new ArrayDeque<>(); // 创建队列
        q.offer(root); // 添加根节点
        while (!q.isEmpty()) { // 如果队列不为空
            ans.add(q.peekLast().val); // 添加右视图的值
            for (int k = q.size(); k > 0; --k) { // 遍历队列
                TreeNode node = q.poll(); // 获取队列中的节点
                if (node.right != null) { // 如果右子树不为空
                    q.offer(node.right); // 添加右子树
                }
                if (node.left != null) { // 如果左子树不为空
                    q.offer(node.left); // 添加左子树
                }
            }
        }
        return ans; // 返回右视图的值
    }

    // 二叉树的层平均值
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> ans = new ArrayList<>(); // 存储每层平均值的列表
        Deque<TreeNode> q = new ArrayDeque<>(); // 创建队列
        q.offer(root); // 添加根节点
        while (!q.isEmpty()) { // 如果队列不为空
            int n = q.size(); // 当前层的节点个数
            long s = 0; // 存储每层节点值的和
            for (int i = 0; i < n; ++i) { // 遍历当前层的节点
                root = q.pollFirst(); // 获取队列中的节点
                s += root.val; // 更新每层节点值的和
                if (root.left != null) { // 如果左子树不为空
                    q.offerLast(root.left); // 添加左子树
                }
                if (root.right != null) { // 如果右子树不为空
                    q.offerLast(root.right); // 添加右子树
                }
            }
            ans.add(s * 1.0 / n); // 添加每层平均值
        }
        return ans; // 返回每层平均值的列表
    }

    // 二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>(); // 存储每层节点的列表
        if (root == null) { // 如果根节点为空
            return ans; // 返回空列表
        }
        Deque<TreeNode> q = new ArrayDeque<>(); // 创建队列
        q.offer(root); // 添加根节点
        while (!q.isEmpty()) { // 如果队列不为空
            List<Integer> t = new ArrayList<>(); // 存储当前层的节点值
            for (int n = q.size(); n > 0; --n) { // 遍历当前层的节点
                TreeNode node = q.poll(); // 获取队列中的节点
                t.add(node.val); // 添加当前层的节点值
                if (node.left != null) { // 如果左子树不为空
                    q.offer(node.left); // 添加左子树
                }
                if (node.right != null) { // 如果右子树不为空
                    q.offer(node.right); // 添加右子树
                }
            }
            ans.add(t); // 添加当前层的节点值
        }
        return ans; // 返回每层节点的列表
    }

    // 二叉树的锯齿形层序遍历
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>(); // 存储每层节点的列表
        if (root == null) { // 如果根节点为空
            return ans; // 返回空列表
        }
        Deque<TreeNode> q = new ArrayDeque<>(); // 创建队列
        q.offer(root); // 添加根节点
        boolean left = true; // 是否从左到右遍历
        while (!q.isEmpty()) { // 如果队列不为空
            List<Integer> t = new ArrayList<>(); // 存储当前层的节点值
            for (int n = q.size(); n > 0; --n) { // 遍历当前层的节点
                TreeNode node = q.poll(); // 获取队列中的节点
                t.add(node.val); // 添加当前层的节点值
                if (node.left != null) { // 如果左子树不为空
                    q.offer(node.left); // 添加左子树
                }
                if (node.right != null) { // 如果右子树不为空
                    q.offer(node.right); // 添加右子树
                }
            }
            if (!left) { // 如果从左到右遍历
                Collections.reverse(t); // 反转当前层的节点值
            }
            ans.add(t); // 添加当前层的节点值
            left = !left; // 更新是否从左到右遍历
        }
        return ans; // 返回每层节点的列表
    }

    // 二叉搜索树的最小绝对差
    private final int inf = 1 << 30; // 无穷大
    private int ans2 = inf; // 最小差值
    private int pre = -inf; // 前一个节点的值

    public int getMinimumDifference(TreeNode root) {
        dfs4(root); // 递归判断
        return ans; // 返回最小差值
    }

    private void dfs4(TreeNode root) {
        if (root == null) { // 如果根节点为空
            return; // 返回
        }
        dfs4(root.left); // 递归判断左子树
        ans2 = Math.min(ans2, root.val - pre); // 更新最小差值
        pre = root.val; // 更新前一个节点的值
        dfs4(root.right); // 递归判断右子树
    }

    // 二叉搜索树中第K小的元素
    public int kthSmallest(TreeNode root, int k) {
        Deque<TreeNode> stk = new ArrayDeque<>(); // 创建栈
        while (root != null || !stk.isEmpty()) { // 如果根节点不为空或栈不为空
            if (root != null) { // 如果根节点不为空
                stk.push(root); // 添加根节点
                root = root.left; // 更新根节点
            } else { // 如果根节点为空
                root = stk.pop(); // 获取栈中的节点
                if (--k == 0) { // 如果k等于0
                    return root.val; // 返回根节点的值
                }
                root = root.right; // 更新根节点
            }
        }
        return -1; // 返回-1
    }

    // 验证二叉搜索树
    private TreeNode prev; // 前一个节点

    public boolean isValidBST(TreeNode root) {
        return dfs5(root); // 递归判断
    }

    private boolean dfs5(TreeNode root) {
        if (root == null) { // 如果根节点为空
            return true; // 返回true
        }
        if (!dfs5(root.left)) { // 如果左子树不为空
            return false; // 返回false
        }
        if (prev != null && prev.val >= root.val) { // 如果前一个节点的值大于等于根节点的值
            return false; // 返回false
        }
        prev = root; // 更新前一个节点
        return dfs5(root.right); // 递归判断右子树
    }
}

// 图
// 岛屿数量
class Solution3 {
    private char[][] grid; // 网格
    private int m; // 行数
    private int n; // 列数

    public int numIslands(char[][] grid) { // 岛屿数量
        this.grid = grid; // 初始化网格
        m = grid.length; // 行数
        n = grid[0].length; // 列数
        int ans = 0; // 岛屿数量
        for (int i = 0; i < m; ++i) { // 遍历行
            for (int j = 0; j < n; ++j) { // 遍历列
                if (grid[i][j] == '1') { // 如果当前位置为陆地
                    dfs(i, j); // 深度优先搜索
                    ++ans; // 岛屿数量加1
                }
            }
        }
        return ans; // 返回岛屿数量
    }

    private void dfs(int i, int j) { // 深度优先搜索
        grid[i][j] = '0'; // 将当前位置标记为水
        int[] dirs = { -1, 0, 1, 0, -1 }; // 方向数组
        for (int k = 0; k < 4; ++k) { // 遍历4个方向
            int x = i + dirs[k]; // 计算下一个位置的行坐标
            int y = j + dirs[k + 1]; // 计算下一个位置的列坐标
            if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == '1') { // 如果下一个位置在网格内且为陆地
                dfs(x, y); // 递归搜索下一个位置
            }
        }
    }

    // 被围绕的区域
    private final int[] dirs = { -1, 0, 1, 0, -1 }; // 方向数组
    private char[][] board; // 网格
    private int m2; // 行数
    private int n2; // 列数

    public void solve(char[][] board) {
        this.board = board; // 初始化网格
        m2 = board.length; // 行数
        n2 = board[0].length; // 列数
        for (int i = 0; i < m2; ++i) { // 遍历行
            dfs2(i, 0); // 深度优先搜索
            dfs2(i, n2 - 1); // 深度优先搜索
        }
        for (int j = 0; j < n2; ++j) { // 遍历列
            dfs2(0, j); // 深度优先搜索
            dfs2(m2 - 1, j); // 深度优先搜索
        }
        for (int i = 0; i < m2; ++i) { // 遍历行
            for (int j = 0; j < n2; ++j) { // 遍历列
                if (board[i][j] == '.') { // 如果当前位置为'.'
                    board[i][j] = 'O'; // 将当前位置标记为'O'
                } else if (board[i][j] == 'O') { // 如果当前位置为'O'
                    board[i][j] = 'X'; // 将当前位置标记为'X'
                }
            }
        }
    }

    private void dfs2(int i, int j) { // 深度优先搜索
        if (i < 0 || i >= m2 || j < 0 || j >= n2 || board[i][j] != 'O') { // 如果当前位置在网格外或当前位置为'O'
            return; // 返回
        }
        board[i][j] = '.'; // 将当前位置标记为'.'
        for (int k = 0; k < 4; ++k) { // 遍历4个方向
            dfs2(i + dirs[k], j + dirs[k + 1]); // 递归搜索下一个位置
        }
    }

    // 克隆图
    class Node {
        public int val; // 节点值
        public List<Node> neighbors; // 邻居节点

        public Node() {
            val = 0; // 节点值
            neighbors = new ArrayList<>(); // 邻居节点
        }

        public Node(int _val) {
            val = _val; // 节点值
            neighbors = new ArrayList<>(); // 邻居节点
        }

        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val; // 节点值
            neighbors = _neighbors; // 邻居节点
        }
    }

    private Map<Node, Node> visited = new HashMap<>(); // 访问过的节点

    public Node cloneGraph(Node node) {
        if (node == null) { // 如果节点为空
            return null; // 返回null
        }
        if (visited.containsKey(node)) { // 如果访问过的节点包含当前节点
            return visited.get(node); // 返回访问过的节点
        }
        Node cloneNode = new Node(node.val, new ArrayList<>()); // 克隆节点
        visited.put(node, cloneNode); // 添加访问过的节点
        for (Node neighbor : node.neighbors) { // 遍历邻居节点
            cloneNode.neighbors.add(cloneGraph(neighbor)); // 递归克隆邻居节点
        }
        return cloneNode; // 返回克隆节点
    }

    // 除法求值
    private Map<String, String> p; // 父节点
    private Map<String, Double> w; // 权重

    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) { // 除法求值
        int n = equations.size(); // 方程数量
        p = new HashMap<>(); // 父节点
        w = new HashMap<>(); // 权重
        for (List<String> e : equations) { // 遍历方程
            p.put(e.get(0), e.get(0)); // 添加父节点
            p.put(e.get(1), e.get(1)); // 添加父节点
            w.put(e.get(0), 1.0); // 添加权重
            w.put(e.get(1), 1.0); // 添加权重
        }
        for (int i = 0; i < n; ++i) { // 遍历方程
            List<String> e = equations.get(i); // 获取方程
            String a = e.get(0), b = e.get(1); // 获取方程的两个变量
            String pa = find(a), pb = find(b); // 获取方程的两个变量的父节点
            if (Objects.equals(pa, pb)) { // 如果两个变量的父节点相同
                continue; // 跳过
            }
            p.put(pa, pb); // 更新父节点
            w.put(pa, w.get(b) * values[i] / w.get(a)); // 更新权重
        }
        int m = queries.size(); // 查询数量
        double[] ans = new double[m]; // 答案
        for (int i = 0; i < m; ++i) { // 遍历查询
            String c = queries.get(i).get(0), d = queries.get(i).get(1); // 获取查询的两个变量
            ans[i] = !p.containsKey(c) || !p.containsKey(d) || !Objects.equals(find(c), find(d)) ? -1.0
                    : w.get(c) / w.get(d); // 计算查询的结果
        }
        return ans; // 返回答案
    }

    private String find(String x) { // 查找父节点
        if (!p.get(x).equals(x)) { // 如果当前节点的父节点不是当前节点
            String origin = p.get(x); // 获取当前节点的父节点
            p.put(x, find(origin)); // 更新当前节点的父节点
            w.put(x, w.get(x) * w.get(origin)); // 更新当前节点的权重
        }
        return p.get(x); // 返回当前节点的父节点
    }

    // 课程表
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<Integer>[] g = new List[numCourses]; // 图
        Arrays.setAll(g, K -> new ArrayList<>()); // 初始化图
        int[] indeg = new int[numCourses]; // 入度
        for (var p : prerequisites) { // 遍历先决条件
            int a = p[0], b = p[1]; // 获取先决条件
            g[b].add(a); // 添加边
            ++indeg[a]; // 增加入度
        }
        Deque<Integer> q = new ArrayDeque<>(); // 队列
        for (int i = 0; i < numCourses; ++i) { // 遍历课程
            if (indeg[i] == 0) { // 如果入度为0
                q.offer(i); // 添加课程
            }
        }
        while (!q.isEmpty()) { // 如果队列不为空
            int i = q.poll(); // 获取课程
            --numCourses; // 减少课程数量
            for (int j : g[i]) { // 遍历邻居节点
                if (--indeg[j] == 0) { // 如果入度为0
                    q.offer(j); // 添加课程
                }
            }
        }
        return numCourses == 0; // 返回是否可以完成课程
    }

    // 课程表II
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        List<Integer>[] g = new List[numCourses]; // 图
        Arrays.setAll(g, K -> new ArrayList<>()); // 初始化图
        int[] indeg = new int[numCourses]; // 入度
        for (var p : prerequisites) { // 遍历先决条件
            int a = p[0], b = p[1]; // 获取先决条件
            g[b].add(a); // 添加边
            ++indeg[a]; // 增加入度
        }
        Deque<Integer> q = new ArrayDeque<>(); // 队列
        for (int i = 0; i < numCourses; ++i) { // 遍历课程
            if (indeg[i] == 0) { // 如果入度为0
                q.offer(i); // 添加课程
            }
        }
        int[] ans = new int[numCourses]; // 答案
        int cnt = 0; // 答案索引
        while (!q.isEmpty()) { // 如果队列不为空
            int i = q.poll(); // 获取课程
            ans[cnt++] = i; // 添加课程
            for (int j : g[i]) { // 遍历邻居节点
                if (--indeg[j] == 0) { // 如果入度为0
                    q.offer(j); // 添加课程
                }
            }
        }
        return cnt == numCourses ? ans : new int[0]; // 返回答案
    }

    // 蛇梯棋
    public int snakesAndLadders(int[][] board) {
        int n = board.length; // 棋盘大小
        Deque<Integer> q = new ArrayDeque<>(); // 队列
        q.offer(1); // 添加起点
        int m = n * n; // 棋盘大小
        boolean[] vis = new boolean[m + 1]; // 访问过的节点
        vis[1] = true; // 标记起点
        for (int ans = 0; !q.isEmpty(); ++ans) { // 如果队列不为空
            for (int k = q.size(); k > 0; --k) { // 遍历队列
                int x = q.poll(); // 获取节点
                if (x == m) { // 如果节点为终点
                    return ans; // 返回答案
                }
                for (int y = x + 1; y <= Math.min(x + 6, m); ++y) { // 遍历6个方向
                    int i = (y - 1) / n, j = (y - 1) % n; // 获取节点行列坐标
                    if (i % 2 == 1) { // 如果行数为奇数
                        j = n - 1 - j; // 反转列数
                    }
                    i = n - i - 1; // 反转行数
                    int z = board[i][j] == -1 ? y : board[i][j]; // 获取节点值
                    if (!vis[z]) { // 如果节点未访问
                        vis[z] = true; // 标记节点
                        q.offer(z); // 添加节点
                    }
                }
            }
        }
        return -1; // 返回-1
    }

    // 最小基因变化
    public int minMutation(String startGene, String endGene, String[] bank) {
        Deque<String> q = new ArrayDeque<>(); // 队列
        q.offer(startGene); // 添加起点
        Set<String> vis = new HashSet<>(); // 访问过的节点
        vis.add(startGene); // 标记起点
        int depth = 0; // 深度
        while (!q.isEmpty()) { // 如果队列不为空
            for (int m = q.size(); m > 0; --m) { // 遍历队列
                String gene = q.poll();
                if (gene.equals(endGene)) { // 如果基因为终点
                    return depth; // 返回答案
                }
                for (String next : bank) { // 遍历基因库
                    int c = 2; // 基因差异
                    for (int k = 0; k < 8 && c > 0; ++k) { // 遍历基因
                        if (gene.charAt(k) != next.charAt(k)) { // 如果基因不同
                            --c; // 减少基因差异
                        }
                    }
                    if (c > 0 && !vis.contains(next)) { // 如果基因差异大于0且未访问
                        vis.add(next); // 标记基因
                        q.offer(next); // 添加基因
                    }
                }
            }
            ++depth; // 增加深度
        }
        return -1; // 返回-1
    }

    // 单词接龙
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> wordSet = new HashSet<>(wordList); // 单词集合
        Queue<String> q = new ArrayDeque<>(); // 队列
        q.offer(beginWord); // 添加起点
        int ans = 1; // 深度
        while (!q.isEmpty()) { // 如果队列不为空
            ++ans;
            for (int i = q.size(); i > 0; --i) { // 遍历队列
                String s = q.poll(); // 获取单词
                char[] chars = s.toCharArray(); // 转换为字符数组
                for (int j = 0; j < chars.length; ++j) { // 遍历字符数组
                    char ch = chars[j]; // 获取字符
                    for (char k = 'a'; k <= 'z'; ++k) { // 遍历字符
                        chars[j] = k; // 更新字符
                        String t = new String(chars); // 转换为字符串
                        if (!wordSet.contains(t)) { // 如果单词集合不包含单词
                            continue; // 跳过
                        }
                        if (endWord.equals(t)) { // 如果单词为终点
                            return ans; // 返回答案
                        }
                        wordSet.remove(t); // 移除单词
                        q.offer(t); // 添加单词
                    }
                    chars[j] = ch; // 恢复字符
                }
            }
        }
        return 0; // 返回0
    }
}

// 实现Trie
class Trie {
    private Trie[] children; // 子节点
    private boolean isEnd; // 是否是结束节点

    public Trie() {
        children = new Trie[26]; // 初始化子节点
    }

    public void insert(String word) {
        Trie node = this; // 当前节点
        for (char ch : word.toCharArray()) { // 遍历单词
            int index = ch - 'a'; // 获取字符索引
            if (node.children[index] == null) { // 如果子节点为空
                node.children[index] = new Trie(); // 创建新节点
            }
            node = node.children[index]; // 更新当前节点
        }
        node.isEnd = true; // 设置结束节点
    }

    public boolean search(String word) {
        Trie node = searchPrefix(word); // 搜索前缀
        return node != null && node.isEnd; // 如果节点不为空且是结束节点
    }

    public boolean startsWith(String prefix) {
        Trie node = searchPrefix(prefix); // 搜索前缀
        return node != null; // 如果节点不为空
    }

    private Trie searchPrefix(String s) {
        Trie node = this; // 当前节点
        for (char ch : s.toCharArray()) { // 遍历字符
            int index = ch - 'a'; // 获取字符索引
            if (node.children[index] == null) { // 如果子节点为空
                return null; // 返回null
            }
            node = node.children[index]; // 更新当前节点
        }
        return node; // 返回当前节点
    }

    // 实现Trie2 支持通配符
    class Trie2 {
        Trie[] children = new Trie[26]; // 子节点
        boolean isEnd; // 是否是结束节点
    }

    class WordDictionary {
        private Trie trie; // 字典树

        public WordDictionary() {
            trie = new Trie(); // 初始化字典树
        }

        public void addWord(String word) {
            Trie node = trie; // 当前节点
            for (char ch : word.toCharArray()) { // 遍历单词
                int index = ch - 'a'; // 获取字符索引
                if (node.children[index] == null) { // 如果子节点为空
                    node.children[index] = new Trie(); // 创建新节点
                }
                node = node.children[index]; // 更新当前节点
            }
            node.isEnd = true; // 设置结束节点
        }

        public boolean search(String word) {
            return search(word, trie); // 搜索单词
        }

        private boolean search(String word, Trie node) {
            for (int i = 0; i < word.length(); ++i) { // 遍历单词
                char ch = word.charAt(i); // 获取字符
                int index = ch - 'a'; // 获取字符索引
                if (ch == '.' && node.children[index] == null) { // 如果字符为通配符且子节点为空
                    return false; // 返回false
                }
                if (ch == '.') { // 如果字符为通配符
                    for (Trie child : node.children) { // 遍历子节点
                        if (child != null && search(word.substring(i + 1), child)) { // 如果子节点不为空且搜索成功
                            return true; // 返回true
                        }
                    }
                    return false; // 返回false
                }
                node = node.children[index]; // 更新当前节点
            }
            return node.isEnd; // 返回是否是结束节点
        }
    }
}

class Solution4 {
    // 单词搜索
    private int m; // 行数
    private int n; // 列数
    private String word; // 单词
    private char[][] board; // 棋盘

    public boolean exist(char[][] board, String word) { // 单词搜索
        m = board.length; // 行数
        n = board[0].length; // 列数
        this.word = word; // 单词
        this.board = board; // 棋盘
        for (int i = 0; i < m; ++i) { // 遍历行
            for (int j = 0; j < n; ++j) { // 遍历列
                if (dfs(i, j, 0)) { // 如果搜索成功
                    return true; // 返回true
                }
            }
        }
        return false; // 返回false
    }

    private boolean dfs(int i, int j, int k) { // 深度优先搜索
        if (k == word.length() - 1) { // 如果单词长度为-1
            return board[i][j] == word.charAt(k); // 如果棋盘字符等于单词字符
        }
        if (board[i][j] != word.charAt(k)) { // 如果棋盘字符不等于单词字符
            return false; // 返回false
        }
        char ch = board[i][j]; // 保存棋盘字符
        board[i][j] = '0'; // 标记为已访问
        int[] dirs = { -1, 0, 1, 0, -1 }; // 方向数组
        for (int u = 0; u < 4; ++u) { // 遍历4个方向
            int x = i + dirs[u], y = j + dirs[u + 1]; // 获取下一个位置
            if (x >= 0 && x < m && y >= 0 && y < n && board[x][y] != '0' && dfs(x, y, k + 1)) { // 如果下一个位置在棋盘内且未访问且搜索成功
                return true; // 返回true
            }
        }
        board[i][j] = ch; // 恢复棋盘字符
        return false; // 返回false
    }

    // 电话号码的字母组合
    public List<String> letterCombinations(String digits) {
        List<String> ans = new ArrayList<>(); // 答案
        if (digits.length() == 0) { // 如果数字长度为0
            return ans; // 返回空列表
        }
        ans.add(""); // 添加空字符串
        String[] d = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" }; // 数字对应的字母
        for (char ch : digits.toCharArray()) { // 遍历数字
            String s = d[ch - '2']; // 获取数字对应的字母
            List<String> t = new ArrayList<>(); // 临时列表
            for (String str : ans) { // 遍历答案
                for (String b : s.split("")) { // 遍历字母
                    t.add(str + b); // 添加字母
                }
            }
            ans = t; // 更新答案
        }
        return ans; // 返回答案
    }

    class Solution5 {
        // 组合
        private List<List<Integer>> ans = new ArrayList<>(); // 答案
        private List<Integer> t = new ArrayList<>(); // 临时列表
        private int n; // 数字长度
        private int k; // 数字个数

        public List<List<Integer>> combine(int n, int k) { // 组合
            this.n = n; // 数字长度
            this.k = k; // 数字个数
            dfs(1); // 深度优先搜索
            return ans; // 返回答案
        }

        private void dfs(int i) { // 深度优先搜索
            if (t.size() == k) { // 如果临时列表长度为k
                ans.add(new ArrayList<>(t)); // 添加临时列表
                return; // 返回
            }
            if (i > n) { // 如果i大于n
                return; // 返回
            }
            t.add(i); // 添加i
            dfs(i + 1); // 深度优先搜索
            t.remove(t.size() - 1); // 移除i
            dfs(i + 1); // 深度优先搜索
        }
    }

    class Solution6 {
        // 组合总和
        private List<List<Integer>> ans = new ArrayList<>(); // 答案
        private List<Integer> t = new ArrayList<>(); // 临时列表
        private boolean[] vis; // 访问数组
        private int[] nums;

        public List<List<Integer>> permute(int[] nums) { // 排列
            this.nums = nums; // 数字
            vis = new boolean[nums.length]; // 访问数组
            dfs(0); // 深度优先搜索
            return ans; // 返回答案
        }

        private void dfs(int i) { // 深度优先搜索
            if (i == nums.length) { // 如果i等于nums长度
                ans.add(new ArrayList<>(t)); // 添加临时列表
                return; // 返回
            }
            for (int j = 0; j < nums.length; ++j) { // 遍历nums
                if (!vis[j]) { // 如果未访问
                    vis[j] = true; // 标记为已访问
                    t.add(nums[j]); // 添加nums[j]
                    dfs(i + 1); // 深度优先搜索
                    t.remove(t.size() - 1); // 移除nums[j]
                    vis[j] = false; // 标记为未访问
                }
            }
        }
    }
}

class Solution7 {
    // 组合总和
    private List<List<Integer>> ans = new ArrayList<>(); // 答案
    private List<Integer> t = new ArrayList<>(); // 临时列表
    private int[] candidates; // 数字

    public List<List<Integer>> combinationSum(int[] candidates, int target) { // 组合总和
        Arrays.sort(candidates); // 排序
        this.candidates = candidates; // 数字
        dfs(0, target); // 深度优先搜索
        return ans; // 返回答案
    }

    private void dfs(int i, int s) { // 深度优先搜索
        if (s == 0) { // 如果s等于0
            ans.add(new ArrayList<>(t)); // 添加临时列表
            return; // 返回
        }
        if (s < candidates[i]) { // 如果s小于candidates[i]
            return; // 返回
        }
        for (int j = i; j < candidates.length; ++j) { // 遍历candidates
            t.add(candidates[j]); // 添加candidates[j]
            dfs(j, s - candidates[j]); // 深度优先搜索
            t.remove(t.size() - 1); // 移除candidates[j]
        }
    }
}

class Solution8 {
    // N皇后 II
    private int n; // 皇后数量
    private int ans; // 答案
    private boolean[] cols = new boolean[10]; // 列
    private boolean[] dg = new boolean[20]; // 主对角线
    private boolean[] udg = new boolean[20]; // 副对角线

    public int totalNQueens(int n) { // N皇后 II
        this.n = n; // 皇后数量
        dfs(0); // 深度优先搜索
        return ans; // 返回答案
    }

    private void dfs(int i) { // 深度优先搜索
        if (i == n) { // 如果i等于n
            ++ans; // 答案加1
            return; // 返回
        }
        for (int j = 0; j < n; ++j) {
            int a = i + j, b = i - j + n;
            if (cols[j] || dg[a] || udg[b]) { // 如果列、主对角线、副对角线被占用
                continue; // 跳过
            }
            cols[j] = true; // 标记为已访问
            dg[a] = true; // 标记为主对角线已访问
            udg[b] = true; // 标记为副对角线已访问
            dfs(i + 1); // 深度优先搜索
            cols[j] = false; // 标记为未访问
            dg[a] = false; // 标记为主对角线未访问
            udg[b] = false; // 标记为副对角线未访问
        }
    }
}

class Solution9 {
    // 括号生成
    private List<String> ans = new ArrayList<>(); // 答案
    private int n; // 括号数量

    public List<String> generateParenthesis(int n) { // 括号生成
        this.n = n; // 括号数量
        dfs(0, 0, ""); // 深度优先搜索
        return ans; // 返回答案
    }

    private void dfs(int i, int j, String s) { // 深度优先搜索
        if (i > n || j > n || i < j) { // 如果i大于n或j大于n或i小于j
            return; // 返回
        }
        if (i == n && j == n) { // 如果i等于n且j等于n
            ans.add(s); // 添加s
            return; // 返回
        }
        dfs(i + 1, j, s + "("); // 深度优先搜索
        dfs(i, j + 1, s + ")"); // 深度优先搜索
    }
}

class Solution10 {
    // 单词搜索
    private int m; // 行数
    private int n; // 列数
    private String word; // 单词
    private char[][] board; // 棋盘

    public boolean exist(char[][] board, String word) { // 单词搜索
        m = board.length; // 行数
        n = board[0].length; // 列数
        this.word = word; // 单词
        this.board = board; // 棋盘
        for (int i = 0; i < m; ++i) { // 遍历行
            for (int j = 0; j < n; ++j) { // 遍历列
                if (dfs(i, j, 0)) { // 如果dfs返回true
                    return true; // 返回true
                }
            }
        }
        return false; // 返回false
    }

    // 单词搜索
    private boolean dfs(int i, int j, int k) { // 深度优先搜索
        if (board[i][j] != word.charAt(k)) { // 如果board[i][j]不等于word[k]
            return false; // 返回false
        }
        if (k == word.length() - 1) { // 如果k等于word长度减1
            return true; // 返回true
        }
        char ch = board[i][j]; // 保存board[i][j]
        board[i][j] = '0'; // 标记为已访问
        int[] dirs = { -1, 0, 1, 0, -1 }; // 结果
        for (int u = 0; u < 4; ++u) { // 遍历dirs
            int x = i + dirs[u], y = j + dirs[u + 1]; // 计算新的坐标
            if (x >= 0 && x < m && y >= 0 && y < n && board[x][y] != '0' && dfs(x, y, k + 1)) { // 如果新的坐标在范围内
                return true; // 返回true
            }
        }
        board[i][j] = ch; // 恢复board[i][j]
        return false; // 返回false
    }
}

// 将有序数组转换为二叉搜索树
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

// 将有序数组转换为二叉搜索树
class Solution11 {
    private int[] nums; // 数组

    public TreeNode sortedArrayToBST(int[] nums) { // 将有序数组转换为二叉搜索树
        this.nums = nums; // 数组
        return dfs(0, nums.length - 1); // 深度优先搜索
    }

    private TreeNode dfs(int left, int right) { // 深度优先搜索
        if (left > right) { // 如果left大于right
            return null; // 返回null
        }
        int mid = (left + right) >> 1; // 计算中间位置
        return new TreeNode(nums[mid], dfs(left, mid - 1), dfs(mid + 1, right)); // 返回新的节点
    }
}

class Solution12 {
    class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    // 排序链表
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) { // 如果head为null或head的下一个节点为null
            return head; // 返回head
        }
        ListNode slow = head, fast = head.next; // 慢指针和快指针
        while (fast != null && fast.next != null) { // 当快指针不为null且快指针的下一个节点不为null
            slow = slow.next; // 慢指针移动
            fast = fast.next.next; // 快指针移动
        }
        ListNode left1 = head, left2 = slow.next; // 左1和左2
        slow.next = null; // 断开链表
        left1 = sortList(left1); // 排序左1
        left2 = sortList(left2); // 排序左2
        ListNode dummy = new ListNode(); // 虚拟头节点
        ListNode tail = dummy; // 尾节点
        while (left1 != null && left2 != null) { // 当左1不为null且左2不为null
            if (left1.val <= left2.val) { // 如果左1的值小于等于左2的值
                tail.next = left1; // 尾节点的下一个节点为左1
                left1 = left1.next; // 左1移动
            } else {
                tail.next = left2; // 尾节点的下一个节点为左2
                left2 = left2.next; // 左2移动
            }
            tail = tail.next; // 尾节点移动
        }
        tail.next = left1 != null ? left1 : left2; // 尾节点的下一个节点为左1或左2
        return dummy.next; // 返回虚拟头节点的下一个节点
    }
}

// 四叉树
class Solution13 {
    class Node { // 节点
        public boolean val; // 值
        public boolean isLeaf; // 是否是叶子节点
        public Node topLeft; // 左上节点
        public Node topRight; // 右上节点
        public Node bottomLeft; // 左下节点
        public Node bottomRight; // 右下节点

        public Node() { // 构造函数
            this.val = false;
            this.isLeaf = false;
            this.topLeft = null;
            this.topRight = null;
            this.bottomLeft = null;
            this.bottomRight = null;
        }

        public Node(boolean val, boolean isLeaf) { // 构造函数
            this.val = val;
            this.isLeaf = isLeaf;
            this.topLeft = null;
            this.topRight = null;
            this.bottomLeft = null;
            this.bottomRight = null;
        }

        public Node(boolean val, boolean isLeaf, Node topLeft, Node topRight, Node bottomLeft, Node bottomRight) { // 构造函数
            this.val = val;
            this.isLeaf = isLeaf;
            this.topLeft = null;
            this.topRight = null;
            this.bottomLeft = null;
            this.bottomRight = null;
        }
    }

    public Node construct(int[][] grid) { // 构造四叉树
        return dfs(0, 0, grid.length - 1, grid.length - 1, grid); // 深度优先搜索
    }

    private Node dfs(int a, int b, int c, int d, int[][] grid) { // 深度优先搜索
        int zero = 0, one = 0; // 0和1
        for (int i = a; i <= c; ++i) { // 遍历行
            for (int j = b; j <= d; ++j) { // 遍历列
                if (grid[i][j] == 0) { // 如果grid[i][j]为0
                    zero = 1; // 0加1
                } else { // 否则
                    one = 1; // 1加1
                }
            }
        }
        boolean isLeaf = zero + one == 1; // 是否是叶子节点
        boolean val = isLeaf && one == 1; // 值
        Node node = new Node(val, isLeaf); // 节点
        if (isLeaf) { // 如果isLeaf为true
            return node; // 返回节点
        }
        node.topLeft = dfs(a, b, (a + c) / 2, (b + d) / 2, grid); // 左上节点
        node.topRight = dfs(a, (b + d) / 2 + 1, (a + c) / 2, d, grid); // 右上节点
        node.bottomLeft = dfs((a + c) / 2 + 1, b, c, (b + d) / 2, grid); // 左下节点
        node.bottomRight = dfs((a + c) / 2 + 1, (b + d) / 2 + 1, c, d, grid); // 右下节点
        return node; // 返回节点
    }
}

// 合并K个升序链表
class Solution14 {
    class ListNode { // 链表节点
        int val; // 值
        ListNode next; // 下一个节点

        ListNode() { // 构造函数
        }

        ListNode(int val) { // 构造函数
            this.val = val;
        }

        ListNode(int val, ListNode next) { // 构造函数
            this.val = val;
            this.next = next;
        }
    }

    public ListNode mergeKLists(ListNode[] lists) { // 合并K个升序链表
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val); // 优先队列
        for (ListNode head : lists) { // 遍历链表
            if (head != null) { // 如果head不为null
                pq.offer(head); // 加入优先队列
            }
        }
        ListNode dummy = new ListNode(); // 虚拟头节点
        ListNode cur = dummy; // 当前节点
        while (!pq.isEmpty()) { // 当优先队列不为空
            ListNode node = pq.poll(); // 取出最小值
            if (node.next != null) { // 如果node的下一个节点不为null
                pq.offer(node.next); // 加入优先队列
            }
            cur.next = node; // 当前节点的下一个节点为node
            cur = cur.next; // 当前节点移动
        }
        return dummy.next; // 返回虚拟头节点的下一个节点
    }
}

// 最大子数组和
class Solution15 {
    public int maxSubArray(int[] nums) {
        int ans = nums[0]; // 最大子数组和
        for (int i = 1, f = nums[0]; i < nums.length; ++i) { // 遍历数组
            f = Math.max(f, 0) + nums[i]; // 更新f
            ans = Math.max(ans, f); // 更新ans
        }
        return ans; // 返回最大子数组和
    }
}

// 最大子数组和II
class Solution16 {
    public int maxSubarraySumCircular(int[] nums) {
        final int inf = 1 << 30; // 无穷大
        int pmi = 0, pmx = -inf; // 最小前缀和和最大前缀和
        int ans = -inf, s = 0, smi = inf; // 最大子数组和
        for (int x : nums) { // 遍历数组
            s += x; // 更新s
            pmi = Math.min(pmi, s); // 更新最小前缀和
            pmx = Math.max(pmx, s); // 更新最大前缀和
            ans = Math.max(ans, s - pmi); // 更新最大子数组和
            smi = Math.min(smi, s - pmx); // 更新最小前缀和
        }
        return ans; // 返回最大子数组和
    }
}

// 搜索插入位置
class Solution17 {
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length; // 左右指针
        while (left < right) { // 当左指针小于等于右指针
            int mid = (left + right) >>> 1; // 中间指针
            if (nums[mid] >= target) { // 如果nums[mid]大于等于target
                right = mid; // 更新右指针
            } else { // 否则
                left = mid + 1; // 更新左指针
            }
        }
        return left; // 返回左指针
    }
}

// 搜索二维矩阵
class Solution18 {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length; // 行数和列数
        int left = 0, right = m * n - 1; // 左右指针
        while (left < right) { // 当左指针小于右指针
            int mid = (left + right) >> 1; // 中间指针
            int x = mid / n, y = mid % n; // 中间值
            if (matrix[x][y] >= target) { // 如果matrix[x][y]大于等于target
                right = mid; // 更新右指针
            } else {
                left = mid + 1; // 更新左指针
            }
        }
        return matrix[left / n][left % n] == target; // 返回matrix[left / n][left % n]是否等于target
    }
}

// 寻找峰值
class Solution19 {
    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1; // 左右指针
        while (left < right) { // 当左指针小于右指针
            int mid = (left + right) >> 1; // 中间指针
            if (nums[mid] > nums[mid + 1]) { // 如果nums[mid]大于nums[mid + 1]
                right = mid; // 更新右指针
            } else {
                left = mid + 1; // 更新左指针
            }
        }
        return left; // 返回左指针
    }
}

// 搜索旋转排序数组
class Solution20 {
    public int search(int[] nums, int target) {
        int n = nums.length; // 数组长度
        int left = 0, right = n - 1; // 左右指针
        while (left < right) { // 当左指针小于右指针
            int mid = (left + right) >> 1; // 中间指针
            if (nums[0] <= nums[mid]) { // 如果nums[0]小于等于nums[mid]
                if (nums[0] <= target && target <= nums[mid]) { // 如果nums[0]小于等于target小于等于nums[mid]
                    right = mid; // 更新右指针
                } else {
                    left = mid + 1; // 更新左指针
                }
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) { // 如果nums[mid]小于target小于等于nums[n - 1]
                    left = mid + 1; // 更新左指针
                } else {
                    right = mid; // 更新右指针
                }
            }
        }
        return nums[left] == target ? left : -1; // 返回nums[left]是否等于target
    }
}

// 在排序数组中查找元素的第一个和最后一个位置
class Solution21 {
    public int[] searchRange(int[] nums, int target) {
        int left = search(nums, target); // 查找target的第一个位置
        int right = search(nums, target + 1); // 查找target + 1的第一个位置
        return 1 == right ? new int[] { -1, -1 } : new int[] { left, right - 1 }; // 返回target的第一个和最后一个位置
    }

    private int search(int[] nums, int x) {
        int left = 0, right = nums.length; // 左右指针
        while (left < right) { // 当左指针小于右指针
            int mid = (left + right) >>> 1; // 中间指针
            if (nums[mid] >= x) { // 如果nums[mid]大于等于x
                right = mid; // 更新右指针
            } else {
                left = mid + 1; // 更新左指针
            }
        }
        return left; // 返回左指针
    }
}

// 寻找旋转排序数组中的最小值
class Solution22 {
    public int findMin(int[] nums) {
        int n = nums.length; // 数组长度
        if (nums[0] <= nums[n - 1]) { // 如果nums[0]小于等于nums[n - 1]
            return nums[0]; // 返回nums[0]
        }
        int left = 0, right = n - 1; // 左右指针
        while (left < right) { // 当左指针小于右指针
            int mid = (left + right) >> 1; // 中间指针
            if (nums[0] <= nums[mid]) { // 如果nums[0]小于等于nums[mid]
                left = mid + 1; // 更新左指针
            } else {
                right = mid; // 更新右指针
            }
        }
        return nums[left]; // 返回nums[left]
    }
}

// 寻找两个正序数组的中位数
class Solution23 {
    private int m; // 数组1的长度
    private int n; // 数组2的长度
    private int[] nums1; // 数组1
    private int[] nums2; // 数组2

    public double findMedianSortedArrays(int[] nums1, int[] nums2) { // 寻找两个正序数组的中位数
        m = nums1.length;
        n = nums2.length;
        this.nums1 = nums1;
        this.nums2 = nums2;
        int a = f(0, 0, (m + n + 1) / 2); // 计算中位数 1
        int b = f(0, 0, (m + n + 2) / 2); // 计算中位数 2
        return (a + b) / 2.0; // 返回中位数 1 和 2 的平均值
    }

    private int f(int i, int j, int k) { // 计算中位数
        if (i >= m) { // 如果i大于等于m
            return nums2[j + k - 1]; // 返回nums2[j + k - 1]
        }
        if (j >= n) { // 如果j大于等于n
            return nums1[i + k - 1]; // 返回nums1[i + k - 1]
        }
        if (k == 1) { // 如果k等于1
            return Math.min(nums1[i], nums2[j]); // 返回nums1[i]和nums2[j]的最小值
        }
        int p = k / 2; // 计算p
        int x = i + p - 1 < m ? nums1[i + p - 1] : 1 << 30; // 计算x
        int y = j + p - 1 < n ? nums2[j + p - 1] : 1 << 30; // 计算y
        return x < y ? f(i + p, j, k - p) : f(i, j + p, k - p); // 返回x < y ? f(i + p, j, k - p) : f(i, j + p, k - p)
    }
}

// 数组中的第K个最大元素
class Solution24 {
    private int[] nums; // 数组
    private int k; // 第k个最大元素

    public int findKthLargest(int[] nums, int k) { // 寻找数组中的第k个最大元素
        this.nums = nums;
        this.k = k;
        return quickSelect(0, nums.length - 1);
    }

    private int quickSelect(int left, int right) { // 快速选择
        if (left == right) { // 如果left等于right
            return nums[left]; // 返回nums[left]
        }
        int i = left - 1, j = right + 1; // 初始化i和j
        int x = nums[(left + right) >>> 1]; // 初始化x
        while (i < j) { // 当i小于j
            while (nums[++i] < x) { // 当nums[++i]小于x
            }
            while (nums[--j] > x) { // 当nums[--j]大于x
            }
            if (i < j) { // 如果i小于j
                int temp = nums[i]; // 交换nums[i]和nums[j]
                nums[i] = nums[j]; // 交换nums[i]和nums[j]
                nums[j] = temp; // 交换nums[i]和nums[j]
            }
        }
        if (j < k) { // 如果j小于k
            return quickSelect(j + 1, right); // 递归调用quickSelect(j + 1, right)
        }
        return quickSelect(left, j); // 递归调用quickSelect(left, j)
    }
}

// 最大资本IPO
class Solution25 {
    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        int n = capital.length; // 数组长度
        PriorityQueue<int[]> q1 = new PriorityQueue<>((a, b) -> a[0] - b[0]); // 最小堆
        for (int i = 0; i < n; ++i) { // 遍历数组
            q1.offer(new int[] { capital[i], profits[i] }); // 加入最小堆
        }
        PriorityQueue<Integer> q2 = new PriorityQueue<>((a, b) -> b - a); // 最大堆
        while (k-- > 0) { // 遍历k
            while (!q1.isEmpty() && q1.peek()[0] <= w) { // 当q1不为空且q1的第一个元素的第一个元素小于等于w
                q2.offer(q1.poll()[1]); // 加入最大堆
            }
            if (q2.isEmpty()) { // 当q2为空
                break; // 退出循环
            }
            w += q2.poll(); // 更新w
        }
        return w; // 返回w
    }
}

// 和最小的k个数对
class Solution26 {
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<int[]> q = new PriorityQueue<>(Comparator.comparingInt(a -> a[0])); // 最小堆
        for (int i = 0; i < Math.min(nums1.length, k); ++i) { // 遍历nums1
            q.offer(new int[] { nums1[i] + nums2[0], i, 0 }); // 加入最小堆
        }
        List<List<Integer>> ans = new ArrayList<>(); // 答案
        while (!q.isEmpty() && k > 0) { // 当q不为空且k大于0
            int[] e = q.poll(); // 取出最小堆的第一个元素
            ans.add(Arrays.asList(nums1[e[1]], nums2[e[2]])); // 加入答案
            --k; // 更新k
            if (e[2] + 1 < nums2.length) { // 如果e[2] + 1小于nums2的长度
                q.offer(new int[] { nums1[e[1]] + nums2[e[2] + 1], e[1], e[2] + 1 }); // 加入最小堆
            }
        }
        return ans; // 返回答案
    }
}

// 数据流的中位数
class MedianFinder {
    private PriorityQueue<Integer> minQ = new PriorityQueue<>(); // 最小堆
    private PriorityQueue<Integer> maxQ = new PriorityQueue<>(Collections.reverseOrder()); // 最大堆

    public MedianFinder() { // 构造函数
    }

    public void addNum(int num) { // 添加数字
        maxQ.offer(num); // 加入最小堆
        minQ.offer(maxQ.poll()); // 加入最大堆
        if (minQ.size() - maxQ.size() > 1) { // 如果最小堆的大小小于最大堆的大小
            maxQ.offer(minQ.poll()); // 加入最大堆
        }
    }

    public double findMedian() { // 查找中位数
        return minQ.size() == maxQ.size() ? (minQ.peek() + maxQ.peek()) / 2.0 : minQ.peek(); // 返回最小堆和最大堆的平均值
    }
}

// 二进制求和
class Solution27 {
    public String addBinary(String a, String b) {
        var sb = new StringBuilder(); // 创建一个StringBuilder
        int i = a.length() - 1, j = b.length() - 1; // 初始化i和j
        for (int carry = 0; i >= 0 || j >= 0 || carry > 0; --i, --j) { // 遍历i和j
            carry += (i >= 0 ? a.charAt(i) - '0' : 0) + (j >= 0 ? b.charAt(j) - '0' : 0); // 更新carry
            sb.append(carry % 2); // 添加carry % 2
            carry /= 2; // 更新carry
        }
        return sb.reverse().toString(); // 返回sb的反向字符串
    }
}

// 颠倒二进制位
class Solution28 {
    public int reverseBits(int n) {
        int ans = 0; // 初始化ans
        for (int i = 0; i < 32 && n != 0; ++i) { // 遍历32位
            ans |= (n & 1) << (31 - i); // 更新ans
            n >>>= 1; // 更新n
        }
        return ans; // 返回ans
    }
}

// 位1的个数
class Solution29 {
    public int hammingWeight(int n) {
        int ans = 0; // 初始化ans
        while (n != 0) { // 当n不为0
            n &= n - 1; // 更新n
            ++ans; // 更新ans
        }
        return ans; // 返回ans
    }
}

// 只出现一次的数字
class Solution30 {
    public int singleNumber(int[] nums) {
        int ans = 0; // 初始化ans
        for (int v : nums) { // 遍历nums
            ans ^= v; // 更新ans
        }
        return ans; // 返回ans
    }
}

// 只出现一次的数字 II
class Solution31 {
    public int singleNumber(int[] nums) {
        int ans = 0; // 初始化ans
        for (int i = 0; i < 32; i++) { // 遍历32位
            int cnt = 0; // 初始化cnt
            for (int num : nums) { // 遍历nums
                cnt += num >> i & 1; // 更新cnt
            }
            cnt %= 3; // 更新cnt
            ans |= cnt << 1; // 更新ans
        }
        return ans; // 返回ans
    }
}

// 数字范围按位与
class Solution32 {
    public int rangeBitwiseAnd(int left, int right) {
        while (left < right) { // 当left小于right
            right &= right - 1; // 更新right
        }
        return right; // 返回right
    }
}

// 回文数
class Solution33 {
    public boolean isPalindrome(int x) {
        if (x < 0 || (x > 0 && x % 10 == 0)) { // 如果x小于0或者x大于0且x的个位数为0
            return false; // 返回false
        }
        int y = 0; // 初始化y
        for (; y < x; x /= 10) { // 当y小于x
            y = y * 10 + x % 10; // 更新y
        }
        return x == y || x == y / 10; // 返回x是否等于y或者x是否等于y除以10
    }
}

// 加1
class Solution34 {
    public int[] plusOne(int[] digits) {
        int n = digits.length; // 数组长度
        for (int i = n - 1; i >= 0; --i) { // 遍历数组
            ++digits[i]; // 更新digits[i]
            digits[i] %= 10; // 更新digits[i]
            if (digits[i] != 0) { // 如果digits[i]不等于0
                return digits; // 返回digits
            }
        }
        digits = new int[n + 1]; // 创建一个新数组
        digits[0] = 1; // 更新digits[0]
        return digits; // 返回digits
    }
}

// 尾随零
class Solution35 {
    public int trailingZeroes(int n) {
        int ans = 0; // 初始化ans
        while (n > 0) { // 当n大于0
            n /= 5; // 更新n
            ans += n; // 更新ans
        }
        return ans; // 返回ans
    }
}

// x的平方根
class Solution36 {
    public int mySqrt(int x) {
        int left = 0, right = x; // 初始化left和right
        while (left < right) { // 当left小于right
            int mid = (left + right + 1) >>> 1; // 计算mid
            if (mid > x / mid) { // 如果mid大于x除以mid
                right = mid - 1; // 更新right
            } else {
                left = mid; // 更新left
            }
        }
        return left; // 返回left
    }
}

// pow(x, n)
class Solution37 {
    public double myPow(double x, int n) {
        return n >= 0 ? qpow(x, n) : 1.0 / qpow(x, -(long) n); // 返回n >= 0 ? qpow(x, n) : 1.0 / qpow(x, -(long) n)
    }

    private double qpow(double a, long n) {
        double ans = 1.0; // 初始化ans
        for (; n > 0; n >>= 1) { // 当n大于0
            if ((n & 1) == 1) { // 如果n的最后一位为1
                ans = ans * a; // 更新ans
            }
            a = a * a; // 更新a
        }
        return ans; // 返回ans
    }
}

// 直线上最多的点数
class Solution38 {
    public int maxPoints(int[][] points) {
        int n = points.length; // 数组长度
        int ans = 1; // 初始化ans
        for (int i = 0; i < n; ++i) { // 遍历数组
            int x1 = points[i][0], y1 = points[i][1]; // 初始化x1和y1
            for (int j = i + 1; j < n; ++j) { // 遍历数组
                int x2 = points[j][0], y2 = points[j][1]; // 初始化x2和y2
                int cnt = 2; // 初始化cnt
                for (int k = j + 1; k < n; ++k) { // 遍历数组
                    int x3 = points[k][0], y3 = points[k][1]; // 初始化x3和y3
                    int a = (y2 - y1) * (x3 - x1); // 计算a
                    int b = (y3 - y1) * (x2 - x1); // 计算b
                    if (a == b) { // 如果a等于b
                        ++cnt; // 更新cnt
                    }
                }
                ans = Math.max(ans, cnt); // 更新ans
            }
        }
        return ans; // 返回ans
    }
}

// 爬楼梯
class Solution39 {
    public int climbStairs(int n) {
        int a = 0, b = 1; // 初始化a和b
        for (int i = 0; i < n; ++i) { // 遍历n
            int c = a + b; // 计算c
            a = b; // 更新a
            b = c; // 更新b
        }
        return b; // 返回b
    }
}

// 打家劫舍
class Solution40 {
    private Integer[] f; // 动态规划数组
    private int[] nums; // 数组

    public int rob(int[] nums) {
        this.nums = nums; // 初始化nums
        f = new Integer[nums.length]; // 初始化f
        return dfs(0); // 返回dfs(0)
    }

    private int dfs(int i) {
        if (i >= nums.length) { // 如果i大于等于nums的长度
            return 0; // 返回0
        }
        if (f[i] == null) { // 如果f[i]为null
            f[i] = Math.max(dfs(i + 1), dfs(i + 2) + nums[i]); // 更新f[i]
        }
        return f[i]; // 返回f[i]
    }
}

// 拆分单词
class Solution41 {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> words = new HashSet<>(wordDict); // 创建一个set
        int n = s.length(); // 字符串长度
        boolean[] f = new boolean[n + 1]; // 动态规划数组
        f[0] = true; // 初始化f[0]
        for (int i = 1; i <= n; ++i) { // 遍历n
            for (int j = 0; j < i; ++j) { // 遍历j
                if (f[j] && words.contains(s.substring(j, i))) { // 如果f[j]为true且s.substring(j, i)在words中
                    f[i] = true; // 更新f[i]
                    break; // 退出循环
                }
            }
        }
        return f[n]; // 返回f[n]
    }
}

// 零钱兑换
class Solution42 {
    public int coinChange(int[] coins, int amount) {
        final int inf = 1 << 30; // 初始化inf
        int m = coins.length; // 数组长度
        int n = amount; // 金额
        int[][] f = new int[m + 1][n + 1]; // 动态规划数组
        for (var g : f) {
            Arrays.fill(g, inf); // 填充f
        }
        f[0][0] = 0; // 初始化f[0][0]
        for (int i = 1; i <= m; ++i) { // 遍历m
            for (int j = 0; j <= n; ++j) { // 遍历n
                f[i][j] = f[i - 1][j]; // 更新f[i][j]
                if (j >= coins[i - 1]) { // 如果j大于等于coins[i - 1]
                    f[i][j] = Math.min(f[i][j], f[i][j - coins[i - 1]] + 1); // 更新f[i][j]
                }
            }
        }
        return f[m][n] == inf ? -1 : f[m][n]; // 返回f[m][n]
    }
}

// 最长上升子序列
class Solution43 {
    public int lengthOfLIS(int[] nums) {
        int n = nums.length; // 数组长度
        int[] f = new int[n]; // 动态规划数组
        Arrays.fill(f, 1); // 填充f
        int ans = 1; // 初始化ans
        for (int i = 1; i < n; ++i) { // 遍历n
            for (int j = 0; j < i; ++j) { // 遍历j
                if (nums[j] < nums[i]) { // 如果nums[j]小于nums[i]
                    f[i] = Math.max(f[i], f[j] + 1); // 更新f[i]
                }
            }
            ans = Math.max(ans, f[i]); // 更新ans
        }
        return ans; // 返回ans
    }
}

// 三角形最小路径和
class Solution44 {
    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size(); // 数组长度
        int[] f = new int[n + 1]; // 动态规划数组
        for (int i = n - 1; i >= 0; --i) { // 遍历n
            for (int j = 0; j < i; ++j) { // 遍历j
                f[j] = Math.min(f[j], f[j + 1]) + triangle.get(i).get(j); // 更新f[j]
            }
        }
        return f[0]; // 返回f[0]
    }
}

// 最小路径和
class Solution45 {
    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length; // 数组长度
        int[][] f = new int[m][n]; // 动态规划数组
        f[0][0] = grid[0][0]; // 初始化f[0][0]
        for (int i = 1; i < m; ++i) { // 遍历m
            f[i][0] = f[i - 1][0] + grid[i][0]; // 更新f[i][0]
        }
        for (int j = 1; j < n; ++j) { // 遍历n
            f[0][j] = f[0][j - 1] + grid[0][j]; // 更新f[0][j]
        }
        for (int i = 1; i < m; ++i) { // 遍历m
            for (int j = 1; j < n; ++j) { // 遍历n
                f[i][j] = Math.min(f[i - 1][j], f[i][j - 1]) + grid[i][j]; // 更新f[i][j]
            }
        }
        return f[m - 1][n - 1]; // 返回f[m - 1][n - 1]
    }
}

// 不同路径 II
class Solution46 {
    private Integer[][] f; // 动态规划数组
    private int m; // 行数
    private int n; // 列数
    private int[][] obstacleGrid; // 障碍物网格

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        m = obstacleGrid.length; // 行数
        n = obstacleGrid[0].length; // 列数
        f = new Integer[m][n]; // 动态规划数组
        this.obstacleGrid = obstacleGrid; // 障碍物网格
        return dfs(0, 0); // 返回dfs(0, 0)
    }

    private int dfs(int i, int j) {
        if (i >= m || j >= n || obstacleGrid[i][j] == 1) { // 如果i大于等于m或j大于等于n或obstacleGrid[i][j]为1
            return 0; // 返回0
        }
        if (i == m - 1 && j == n - 1) { // 如果i等于m - 1且j等于n - 1
            return 1; // 返回1
        }
        if (f[i][j] == null) { // 如果f[i][j]为null
            f[i][j] = dfs(i + 1, j) + dfs(i, j + 1); // 更新f[i][j]
        }
        return f[i][j]; // 返回f[i][j]
    }
}

// 最长回文子串
class Solution47 {
    public String longestPalindrome(String s) {
        int n = s.length(); // 字符串长度
        boolean[][] f = new boolean[n][n]; // 动态规划数组
        for (var g : f) { // 遍历f
            Arrays.fill(g, true); // 填充f
        }
        int k = 0, mx = 1; // 初始化k和mx
        for (int i = n - 2; i >= 0; --i) { // 遍历n
            for (int j = i + 1; j < n; ++j) { // 遍历n
                f[i][j] = false; // 更新f[i][j]
                if (s.charAt(i) == s.charAt(j)) { // 如果s.charAt(i)等于s.charAt(j)
                    f[i][j] = f[i + 1][j - 1]; // 更新f[i][j]
                    if (f[i][j] && mx < j - i + 1) { // 如果f[i][j]为true且mx小于j - i + 1
                        mx = j - i + 1; // 更新mx
                        k = i; // 更新k
                    }
                }
            }
        }
        return s.substring(k, k + mx); // 返回s.substring(k, k + mx)
    }
}

// 交错字符串
class Solution48 {
    private Map<List<Integer>, Boolean> f = new HashMap<>(); // 记忆化搜索
    private String s1; // 字符串1
    private String s2; // 字符串2
    private String s3; // 字符串3
    private int m; // 字符串1的长度
    private int n; // 字符串2的长度

    public boolean isInterleave(String s1, String s2, String s3) {
        m = s1.length(); // 字符串1的长度
        n = s2.length(); // 字符串2的长度
        if (m + n != s3.length()) { // 如果字符串1的长度加上字符串2的长度不等于字符串3的长度
            return false; // 返回false
        }
        this.s1 = s1; // 字符串1
        this.s2 = s2; // 字符串2
        this.s3 = s3; // 字符串3
        return dfs(0, 0); // 返回dfs(0, 0)
    }

    private boolean dfs(int i, int j) {
        if (i >= m && j >= n) { // 如果i大于等于m且j大于等于n
            return true; // 返回true
        }
        var key = List.of(i, j); // 创建一个列表
        if (f.containsKey(key)) { // 如果f包含key
            return f.get(key); // 返回f.get(key)
        }
        int k = i + j; // 计算k
        boolean ans = false; // 初始化ans
        if (i < m && s1.charAt(i) == s3.charAt(k) && dfs(i + 1, j)) { // 如果i小于m且s1.charAt(i)等于s3.charAt(k)且dfs(i + 1,
                                                                      // j)为true
            ans = true; // 更新ans
        }
        if (!ans && j < n && s2.charAt(j) == s3.charAt(k) && dfs(i, j + 1)) { // 如果ans为false且j小于n且s2.charAt(j)等于s3.charAt(k)且dfs(i,
                                                                              // j + 1)为true
            ans = true; // 更新ans
        }
        f.put(key, ans); // 更新f
        return ans; // 返回ans
    }
}

// 编辑距离
class Solution49 {
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length(); // 字符串1的长度和字符串2的长度
        int[][] f = new int[m + 1][n + 1]; // 动态规划数组
        for (int j = 1; j <= n; ++j) { // 遍历n
            f[0][j] = j; // 更新f[0][j]
        }
        for (int i = 1; i <= m; ++i) { // 遍历m
            f[i][0] = i; // 更新f[i][0]
            for (int j = 1; j <= n; ++j) { // 遍历n
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) { // 如果word1.charAt(i - 1)等于word2.charAt(j - 1)
                    f[i][j] = f[i - 1][j - 1]; // 更新f[i][j]
                } else {
                    f[i][j] = Math.min(f[i - 1][j], Math.min(f[i][j - 1], f[i - 1][j - 1])) + 1; // 更新f[i][j]
                }
            }
        }
        return f[m][n]; // 返回f[m][n]
    }
}

// 买卖股票的最佳时机 III
class Solution50 {
    public int maxProfit(int[] prices) {
        int f1 = -prices[0], f2 = 0, f3 = -prices[0], f4 = 0; // 初始化f1, f2, f3, f4
        for (int i = 1; i < prices.length; ++i) { // 遍历prices
            f1 = Math.max(f1, -prices[i]); // 更新f1
            f2 = Math.max(f2, f1 + prices[i]); // 更新f2
            f3 = Math.max(f3, f2 - prices[i]); // 更新f3
            f4 = Math.max(f4, f3 + prices[i]); // 更新f4
        }
        return f4; // 返回f4
    }
}

// 买卖股票的最佳时机 IV
class Solution51 {
    private Integer[][][] f; // 动态规划数组
    private int[] prices; // 价格数组
    private int n; // 最大交易次数

    public int maxProfit(int k, int[] prices) {
        n = prices.length; // 数组长度
        this.prices = prices; // 价格数组
        f = new Integer[n][k + 1][2];
        return dfs(0, k, 0);
    }

    private int dfs(int i, int j, int k) {
        if (i >= n) { // 如果i大于等于n或k为0
            return 0; // 返回0
        }
        if (f[i][j][k] != null) { // 如果f[i][j][k]为null
            return f[i][j][k]; // 返回f[i][j][k]
        }
        int ans = dfs(i + 1, j, k); // 更新ans
        if (k > 0) { // 如果k大于0
            ans = Math.max(ans, prices[i] + dfs(i + 1, j, 0)); // 更新ans
        } else if (j > 0) { // 如果j大于0
            ans = Math.max(ans, -prices[i] + dfs(i + 1, j - 1, 1)); // 更新ans
        }
        return f[i][j][k] = ans; // 返回f[i][j][k]
    }
}

// 最大正方形
class Solution52 {
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length, n = matrix[0].length; // 矩阵的行数和列数
        int[][] dp = new int[m + 1][n + 1]; // 动态规划数组
        int mx = 0; // 最大正方形边长
        for (int i = 0; i < m; ++i) { // 遍历m
            for (int j = 0; j < n; ++j) { // 遍历n
                if (matrix[i][j] == '1') { // 如果matrix[i][j]为'1'
                    dp[i + 1][j + 1] = Math.min(Math.min(dp[i][j + 1], dp[i + 1][j]), dp[i][j]) + 1; // 更新dp[i + 1][j +
                                                                                                     // 1]
                    mx = Math.max(mx, dp[i + 1][j + 1]); // 更新mx
                }
            }
        }
        return mx * mx; // 返回mx * mx
    }
}