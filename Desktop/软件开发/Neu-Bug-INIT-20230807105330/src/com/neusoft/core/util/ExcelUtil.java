package com.neusoft.core.util;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletResponse;

import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRichTextString;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;

/**
 * @ClassName: ExcelUtil
 * @Description: excel工具类
 */
public class ExcelUtil {

	/**
	 * @Title: export
	 * @Description: 导出数据到excel方法
	 * @param filename
	 *            保存到客户端的文件名
	 * @param title
	 *            标题行数组，例：String[]{"名称","地址"}
	 * @param key
	 *            从查询结果List取得的MAP的KEY顺序，需要和title顺序匹配，例：String[]{"name","address"
	 *            }
	 * @param values
	 *            结果集
	 * @param httpServletResponse
	 * @throws IOException
	 */
	@SuppressWarnings({ "rawtypes", "deprecation" })
	public static void export(String filename, String[] title, String[] key,
			List<Map> values, HttpServletResponse httpServletResponse)
			throws IOException {
		String filename2 = new String(filename.getBytes(), "iso-8859-1");
		ServletOutputStream servletOutputStream = httpServletResponse
				.getOutputStream();
		HSSFWorkbook workbook = null;
		httpServletResponse.setHeader("Content-disposition",
				"attachment; filename=" + filename2);
		httpServletResponse.setContentType("application/x-download");
		workbook = new HSSFWorkbook();
		HSSFSheet sheet = workbook.createSheet();
		HSSFRow row = null;
		HSSFCell cell = null;
		row = sheet.createRow((short) 0);
		for (int i = 0; title != null && i < title.length; i++) {
			cell = row.createCell((short) i);
			cell.setCellType(HSSFCell.CELL_TYPE_STRING);
			cell.setCellValue(new HSSFRichTextString(title[i]));
		}
		Map map = null;
		for (int i = 0; values != null && i < values.size(); i++) {
			row = sheet.createRow((short) (i + 1));
			map = values.get(i);
			for (int i2 = 0; i2 < key.length; i2++) {
				cell = row.createCell((short) (i2));
				cell.setCellType(HSSFCell.CELL_TYPE_STRING);
				if (map.get(key[i2]) == null) {
					cell.setCellValue(new HSSFRichTextString(""));
				} else {
					cell.setCellValue(new HSSFRichTextString(map.get(key[i2])
							.toString()));
				}
			}
		}
		workbook.write(servletOutputStream);
		servletOutputStream.flush();
		servletOutputStream.close();
	}
}
