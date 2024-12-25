package com.neusoft.core.util;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * html工具类
 */
public class HTMLUtil {

	private final static String CHINESE_PATTERN = "^[\u2E80-\u9FFF，。！？、“；：《》——）（]$";
	private static Pattern pattern;
	private static Matcher matcher;

	/**
	 * 进行HTML的标签过滤，获得纯文本 并且将字符和数字进行2个等同于1个汉字长度来计算
	 * 
	 * @param content
	 *            待处理的内容
	 * @param length
	 *            需要返回的长度
	 * @return String
	 */
	public static String getNoHTMLString(String content, int length) {
		// 考虑到英文和数字，取2倍的长度字符（冗余）
		String dealContent = getNoHTMLStringInner(content, length * 2);
		// 结合英文和数字的长度进行长度动态计算
		dealContent = dealNoChinese(dealContent, length);
		return dealContent;
	}

	/**
	 * 将字符和数字进行2个等同于1个汉字长度来计算（不进行html标签取代）
	 * 
	 * @param content
	 *            待处理的内容
	 * @param length
	 *            需要返回的长度
	 * @return String
	 */
	public static String getSubString(String content, int length) {
		// 结合英文和数字的长度进行长度动态计算
		String dealContent = dealNoChinese(content, length);
		return dealContent;
	}

	/**
	 * 处理非汉字时，处理获取2个字符
	 * 
	 * @param content
	 * @param length
	 * @return String
	 */
	private static String dealNoChinese(String content, int length) {
		if (content == null) {
			return "";
		}
		if (content.length() <= length) {
			return content;
		}

		String returnValue = "";
		float noChineseNum = 0;
		char[] charArr = content.toCharArray();
		for (int i = 0; i < charArr.length; i++) {
			if (isChinese(charArr[i] + "")) {
				noChineseNum++;
			} else {
				noChineseNum += 0.5;
			}
			if (noChineseNum > length) {
				returnValue = content.substring(0, i);
				break;
			}
		}

		if (content.length() > returnValue.length()) {
			if (returnValue.length() == 0) {
				returnValue = content;
			} else {
				returnValue += "...";
			}
		} else {

		}
		return returnValue;
	}

	/**
	 * @Title: isChinese
	 * @Description: 是否为中文
	 * @param str
	 * @return boolean
	 */
	private static boolean isChinese(String str) {
		pattern = Pattern.compile(CHINESE_PATTERN);
		matcher = pattern.matcher(str);
		return matcher.matches();
	}

	/**
	 * @Title: getNoHTMLStringInner
	 * @Description: 得到相应位数已过滤html、script、style 标签的内容 内容结尾 为...
	 * @param content
	 *            内容String
	 * @param p
	 *            位数
	 * @return String
	 */
	private static String getNoHTMLStringInner(String content, int p) {

		if (null == content)
			return "";
		if (0 == p)
			return "";

		Pattern p_script;
		Matcher m_script;
		Pattern p_style;
		Matcher m_style;
		Pattern p_html;
		Matcher m_html;

		try {
			String regEx_script = "<[\\s]*?script[^>]*?>[\\s\\S]*?<[\\s]*?\\/[\\s]*?script[\\s]*?>";
			// 定义script的正则表达式{或<script[^>]*?>[\\s\\S]*?<\\/script> }
			String regEx_style = "<[\\s]*?style[^>]*?>[\\s\\S]*?<[\\s]*?\\/[\\s]*?style[\\s]*?>";
			// 定义style的正则表达式{或<style[^>]*?>[\\s\\S]*?<\\/style> }
			String regEx_html = "<[^>]+>"; // 定义HTML标签的正则表达式

			p_script = Pattern.compile(regEx_script, Pattern.CASE_INSENSITIVE);
			m_script = p_script.matcher(content);
			content = m_script.replaceAll(""); // 过滤script标签
			p_style = Pattern.compile(regEx_style, Pattern.CASE_INSENSITIVE);
			m_style = p_style.matcher(content);
			content = m_style.replaceAll(""); // 过滤style标签

			p_html = Pattern.compile(regEx_html, Pattern.CASE_INSENSITIVE);
			m_html = p_html.matcher(content);

			content = m_html.replaceAll(""); // 过滤html标签
		} catch (Exception e) {
			return "";
		}

		if (content.length() > p) {
			content = content.substring(0, p);
			content = content + "...";
		}
		return content;
	}

}
